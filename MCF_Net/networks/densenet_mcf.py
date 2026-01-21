# encoding: utf-8
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

class DenseNet121_v0(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, n_class):
        super(DenseNet121_v0, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        # Removed activation - CrossEntropyLoss will apply softmax internally
        self.densenet121.classifier = nn.Linear(num_ftrs, n_class)

    def forward(self, x):
        x = self.densenet121(x)
        return x


class dense121_mcs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(dense121_mcs, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features

        A_model = DenseNet121_v0(n_class=n_class)
        self.featureA = A_model
        self.classA = A_model.densenet121.features

        B_model = DenseNet121_v0(n_class=n_class)
        self.featureB = B_model
        self.classB = B_model.densenet121.features

        C_model = DenseNet121_v0(n_class=n_class)
        self.featureC = C_model
        self.classC = C_model.densenet121.features

        # Removed activation - CrossEntropyLoss will apply softmax internally
        self.combine1 = nn.Linear(n_class * 4, n_class)
        self.combine2 = nn.Linear(num_ftrs * 3, n_class)

    def forward(self, x, y, z):
        # Get raw logits from each channel
        x1_raw = self.featureA(x) # inference from backbone A
        y1_raw = self.featureB(y) # inference from backbone B
        z1_raw = self.featureC(z) # inference from backbone C

        x2 = self.classA(x)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)

        y2 = self.classB(y)
        y2 = F.relu(y2, inplace=True)
        y2 = F.adaptive_avg_pool2d(y2, (1, 1)).view(y2.size(0), -1)
        
        z2 = self.classC(z)
        z2 = F.relu(z2, inplace=True)
        z2 = F.adaptive_avg_pool2d(z2, (1, 1)).view(z2.size(0), -1)

        combine = torch.cat((x2.view(x2.size(0), -1),
                             y2.view(y2.size(0), -1),
                             z2.view(z2.size(0), -1)), 1)
        combine_raw = self.combine2(combine)

        # Apply softmax for fusion (internal use only)
        x1_soft = F.softmax(x1_raw, dim=1)
        y1_soft = F.softmax(y1_raw, dim=1)
        z1_soft = F.softmax(z1_raw, dim=1)
        combine_soft = F.softmax(combine_raw, dim=1)

        # Concatenate softmax versions for better semantic fusion
        combine3 = torch.cat((x1_soft.view(x1_soft.size(0), -1),
                              y1_soft.view(y1_soft.size(0), -1),
                              z1_soft.view(z1_soft.size(0), -1),
                              combine_soft.view(combine_soft.size(0), -1)), 1)

        combine3_raw = self.combine1(combine3)

        # Return raw logits for CrossEntropyLoss
        return x1_raw, y1_raw, z1_raw, combine_raw, combine3_raw
