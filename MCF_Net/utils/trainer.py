import time
import torch
from progress.bar import Bar
import numpy as np
import pandas as pd


def train_step(train_loader, model, epoch, optimizer, criterion, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0
    loss_w =args.loss_w

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)

        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        labels = labels.cuda()
        
        # Convert one-hot labels to class indices for CrossEntropyLoss
        labels_indices = torch.argmax(labels, dim=1)

        out_A, out_B, out_C, out_F, combine = model(imagesA, imagesB, imagesC)

        loss_x = criterion(out_A, labels_indices)
        loss_y = criterion(out_B, labels_indices)
        loss_z = criterion(out_C, labels_indices)
        loss_c = criterion(out_F, labels_indices)
        loss_f = criterion(combine, labels_indices)

        lossValue = loss_w[0]*loss_x+loss_w[1]*loss_y+loss_w[2]*loss_z+loss_w[3]*loss_c+loss_w[4]*loss_f


        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time=batch_time*(iters_per_epoch-step)/60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss


def validation_step(val_loader, model, criterion):
    """
    Perform validation step with progress bar.
    
    Returns:
        epoch_loss: Average loss over validation set
        all_preds: Numpy array of predicted class indices
        all_labels: Numpy array of true class indices
        all_probs: Numpy array of prediction probabilities/logits
    """
    # switch to eval mode
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)
    bar.check_tty = False

    with torch.no_grad():
        for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
            start_time = time.time()

            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
            imagesC = imagesC.cuda()
            labels = labels.cuda()
            
            # Convert one-hot labels to class indices for CrossEntropyLoss
            labels_indices = torch.argmax(labels, dim=1)

            _, _, _, _, outputs = model(imagesA, imagesB, imagesC)
            
            # Loss
            loss = criterion(outputs, labels_indices)
            epoch_loss += loss.item()
            
            # Collect predictions and labels
            probs = outputs.cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels_indices.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels_np)
            all_probs.append(probs)

            end_time = time.time()

            # measure elapsed time
            batch_time = end_time - start_time
            bar_str = '{} / {} | Time: {batch_time:.2f} mins'
            bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
            bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    return epoch_loss, all_preds, all_labels, all_probs


def save_output(label_test_file, dataPRED, args, save_file):
    label_list = args.label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["image"].tolist()

    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['image_name'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['image_name']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)


