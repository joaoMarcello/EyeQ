import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score


def compute_metric(datanpGT, datanpPRED, target_names):

    n_class = len(target_names)
    argmaxPRED = np.argmax(datanpPRED, axis=1)
    F1_metric = np.zeros([n_class, 1])
    tn = np.zeros([n_class, 1])
    fp = np.zeros([n_class, 1])
    fn = np.zeros([n_class, 1])
    tp = np.zeros([n_class, 1])

    Accuracy_score = accuracy_score(datanpGT, argmaxPRED)
    ROC_curve = {}
    mAUC = 0

    for i in range(n_class):
        tmp_label = datanpGT == i
        tmp_pred = argmaxPRED == i
        F1_metric[i] = f1_score(tmp_label, tmp_pred)
        tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()
        outAUROC = roc_auc_score(tmp_label, datanpPRED[:, i])

        mAUC = mAUC + outAUROC
        [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, datanpPRED[:, i])

        ROC_curve.update({'ROC_fpr_'+str(i): roc_fpr,
                          'ROC_tpr_' + str(i): roc_tpr,
                          'ROC_T_' + str(i): roc_thresholds,
                          'AUC_' + str(i): outAUROC})

    mPrecision = sum(tp) / sum(tp + fp) if sum(tp + fp) > 0 else 0.0
    mRecall = sum(tp) / sum(tp + fn) if sum(tp + fn) > 0 else 0.0
    mSpecificity = sum(tn) / sum(fp + tn) if sum(fp + tn) > 0 else 0.0
    
    # Avoid division by zero in per-class metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (fp + tn)
        
        # Replace NaN/Inf with 0
        sensitivity = np.nan_to_num(sensitivity, nan=0.0, posinf=0.0, neginf=0.0)
        precision = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)
        specificity = np.nan_to_num(specificity, nan=0.0, posinf=0.0, neginf=0.0)
    
    output = {
        'class_name': target_names,
        'F1': F1_metric,
        'AUC': mAUC / 3,
        'Accuracy': Accuracy_score,

        'Sensitivity': sensitivity,
        'Precision': precision,
        'Specificity': specificity,
        'ROC_curve': ROC_curve,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,

        'micro-Precision': mPrecision,
        'micro-Sensitivity': mRecall,
        'micro-Specificity': mSpecificity,
        'micro-F1': 2*mPrecision * mRecall / (mPrecision + mRecall) if (mPrecision + mRecall) > 0 else 0.0,
    }

    return output
