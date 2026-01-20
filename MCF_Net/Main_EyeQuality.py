import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import train_step, validation_step, save_output
from utils.metric import compute_metric

import pandas as pd
from networks.densenet_mcf import dense121_mcs
from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
import json
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)


def generate_detailed_report(gt_labels, predictions, metrics_dict, args, best_epoch, best_loss, class_names):
    """
    Gera relatório detalhado de métricas no estilo do evaluate.py
    
    Args:
        gt_labels: Array com labels verdadeiros
        predictions: Array com probabilidades preditas (n_samples, n_classes)
        metrics_dict: Dicionário retornado por compute_metric
        args: Argumentos do parser
        best_epoch: Época do melhor modelo
        best_loss: Loss do melhor modelo
        class_names: Lista com nomes das classes ['Good', 'Usable', 'Reject']
    """
    report = []
    
    # Header
    report.append("=" * 116)
    report.append("RELATÓRIO DE AVALIAÇÃO - MCF-NET (ITAPECURU DATASET)")
    report.append("=" * 116)
    report.append(f"\nData/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Modelo: {args.save_model}")
    report.append(f"Melhor época: {best_epoch + 1}/{args.epochs}")
    report.append(f"Melhor validation loss: {best_loss:.4f}")
    
    # Configuration
    report.append("\n" + "=" * 116)
    report.append("CONFIGURAÇÃO DO EXPERIMENTO")
    report.append("=" * 116)
    report.append(f"Dataset: Itapecuru")
    report.append(f"Train CSV: {args.model_dir}/../data/revised/color/itapecuru_train_color_rev.csv")
    report.append(f"Valid CSV: {args.model_dir}/../data/revised/color/itapecuru_valid_color_rev.csv")
    report.append(f"Test CSV:  {args.model_dir}/../data/revised/color/itapecuru_test_color_rev.csv")
    report.append(f"Batch Size: {args.batch_size}")
    report.append(f"Learning Rate: {args.lr}")
    report.append(f"Épocas: {args.epochs}")
    report.append(f"Loss Weights: {args.loss_w}")
    report.append(f"Image Size: {args.crop_size}")
    
    # Compute confusion matrix
    pred_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # Compute per-class metrics from confusion matrix
    n_classes = len(class_names)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    precision_per_class = {}
    recall_per_class = {}
    specificity_per_class = {}
    f1_per_class = {}
    iou_per_class = {}
    support = {}
    
    for i in range(n_classes):
        support[i] = int(cm[i].sum())
        
        # Precision
        precision_per_class[i] = float(tp[i] / (tp[i] + fp[i])) if (tp[i] + fp[i]) > 0 else 0.0
        
        # Recall/Sensitivity
        recall_per_class[i] = float(tp[i] / (tp[i] + fn[i])) if (tp[i] + fn[i]) > 0 else 0.0
        
        # Specificity
        specificity_per_class[i] = float(tn[i] / (tn[i] + fp[i])) if (tn[i] + fp[i]) > 0 else 0.0
        
        # F1
        p = precision_per_class[i]
        r = recall_per_class[i]
        f1_per_class[i] = 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)
        
        # IoU
        denom = tp[i] + fp[i] + fn[i]
        iou_per_class[i] = float(tp[i] / denom) if denom > 0 else 0.0
    
    # Aggregate metrics
    total_samples = len(gt_labels)
    weights = np.array(list(support.values())) / total_samples if total_samples > 0 else np.zeros(n_classes)
    
    precision_macro = float(np.mean(list(precision_per_class.values())))
    precision_weighted = float(np.sum(weights * np.array(list(precision_per_class.values()))))
    
    recall_macro = float(np.mean(list(recall_per_class.values())))
    recall_weighted = float(np.sum(weights * np.array(list(recall_per_class.values()))))
    
    specificity_macro = float(np.mean(list(specificity_per_class.values())))
    specificity_weighted = float(np.sum(weights * np.array(list(specificity_per_class.values()))))
    
    iou_macro = float(np.mean(list(iou_per_class.values())))
    iou_weighted = float(np.sum(weights * np.array(list(iou_per_class.values()))))
    
    f1_macro = float(np.mean(list(f1_per_class.values())))
    f1_weighted = float(np.sum(weights * np.array(list(f1_per_class.values()))))
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(gt_labels, pred_labels)
    
    # Accuracy
    accuracy = float(np.mean(gt_labels == pred_labels))
    
    # AUC from metrics_dict
    auc = float(metrics_dict.get('AUC', 0.0))
    
    # Per-class metrics table
    report.append("\n" + "=" * 116)
    report.append("MÉTRICAS POR CLASSE - TEST SET")
    report.append("=" * 116)
    report.append(f"{'Classe':<12} | {'Suporte':>10} | {'Precision':>12} | {'Recall':>12} | {'Specificity':>13} | {'F1-Score':>12} | {'IoU/Jaccard':>12}")
    report.append("-" * 116)
    
    for i in range(n_classes):
        class_name = class_names[i]
        report.append(
            f"{class_name:<12} | {support[i]:>10} | {precision_per_class[i]:>12.4f} | {recall_per_class[i]:>12.4f} | "
            f"{specificity_per_class[i]:>13.4f} | {f1_per_class[i]:>12.4f} | {iou_per_class[i]:>12.4f}"
        )
    
    report.append("")
    report.append("=" * 116)
    
    # Aggregate metrics table
    report.append("\nMÉTRICAS AGREGADAS - TEST SET")
    report.append("=" * 116)
    report.append(f"{'Métrica':<25} | {'Macro (avg)':>15} | {'Weighted (avg)':>15}")
    report.append("-" * 70)
    
    report.append(f"{'Precision':<25} | {precision_macro:>15.4f} | {precision_weighted:>15.4f}")
    report.append(f"{'Recall (Sensitivity)':<25} | {recall_macro:>15.4f} | {recall_weighted:>15.4f}")
    report.append(f"{'Specificity':<25} | {specificity_macro:>15.4f} | {specificity_weighted:>15.4f}")
    report.append(f"{'F1-Score':<25} | {f1_macro:>15.4f} | {f1_weighted:>15.4f}")
    report.append(f"{'IoU/Jaccard':<25} | {iou_macro:>15.4f} | {iou_weighted:>15.4f}")
    report.append(f"{'Accuracy':<25} | {accuracy:>15.4f} | {accuracy:>15.4f}")
    report.append(f"{'Cohen Kappa':<25} | {kappa:>15.4f} | {kappa:>15.4f}")
    report.append(f"{'AUC':<25} | {auc:>15.4f} | {auc:>15.4f}")
    
    report.append("")
    report.append("=" * 116)
    
    # Confusion matrix
    report.append("\nMATRIZ DE CONFUSÃO - TEST SET")
    report.append("=" * 116)
    report.append("            Predito")
    report.append(f"{'':>16}{'Good':>11}{'Usable':>11}{'Reject':>11}")
    report.append("Real")
    for i, class_name in enumerate(class_names):
        report.append(f"{class_name:<16}{cm[i][0]:>11}{cm[i][1]:>11}{cm[i][2]:>11}")
    
    report.append("")
    report.append("=" * 116)
    
    # Metrics from original compute_metric (for compatibility)
    report.append("\nMÉTRICAS ORIGINAIS (compute_metric)")
    report.append("=" * 116)
    report.append(f"Accuracy: {np.mean(metrics_dict['Accuracy']):.4f}")
    report.append(f"Precision (micro): {float(metrics_dict['micro-Precision']):.4f}")
    report.append(f"Sensitivity (micro): {float(metrics_dict['micro-Sensitivity']):.4f}")
    report.append(f"Specificity (micro): {float(metrics_dict['micro-Specificity']):.4f}")
    report.append(f"F1 (micro): {float(metrics_dict['micro-F1']):.4f}")
    report.append(f"AUC: {metrics_dict['AUC']:.4f}")
    
    report.append("\nPer-class F1 scores:")
    for i, class_name in enumerate(class_names):
        report.append(f"  {class_name}: {float(metrics_dict['F1'][i]):.4f}")
    
    report.append("")
    report.append("=" * 116)
    report.append("FIM DO RELATÓRIO")
    report.append("=" * 116)
    
    return "\n".join(report)


if __name__ == '__main__':
    data_root = r'C:\Users\Public\Documents\DATASETS\Itapecuru\ORIGINAL_CLEAN_768x768'

    # Setting parameters
    parser = argparse.ArgumentParser(description='EyeQ_dense121')
    parser.add_argument('--model_dir', type=str, default='./result/')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--pre_model', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='DenseNet121_Itapecuru')

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])

    parser.add_argument('--n_classes', type=int, default=3)
    # Optimization options
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)

    args = parser.parse_args()

    # Images Labels - Itapecuru dataset
    train_images_dir = data_root  # All images in same directory
    label_train_file = '../data/revised/color/itapecuru_train_color_rev.csv'
    valid_images_dir = data_root  # All images in same directory
    label_valid_file = '../data/revised/color/itapecuru_valid_color_rev.csv'
    test_images_dir = data_root  # All images in same directory
    label_test_file = '../data/revised/color/itapecuru_test_color_rev.csv'

    save_file_name = args.model_dir + args.save_model + '_test_predictions.csv'

    best_metric = np.inf
    best_iter = 0
    # options
    cudnn.benchmark = True

    model = dense121_mcs(n_class=args.n_classes)

    if args.pre_model is not None:
        loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
        model.load_state_dict(loaded_model['state_dict'])

    model.to(device)

    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    transform_list1 = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, +180)),
        ])

    transformList2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    transform_list_val1 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ])

    data_train = DatasetGenerator(data_dir=train_images_dir, list_file=label_train_file, transform1=transform_list1,
                                  transform2=transformList2, n_class=args.n_classes, set_name='train')
    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=0, pin_memory=True)

    data_valid = DatasetGenerator(data_dir=valid_images_dir, list_file=label_valid_file, transform1=transform_list_val1,
                                 transform2=transformList2, n_class=args.n_classes, set_name='valid')
    valid_loader = torch.utils.data.DataLoader(dataset=data_valid, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0, pin_memory=True)

    data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                                 transform2=transformList2, n_class=args.n_classes, set_name='test')
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0, pin_memory=True)

    print(f'\n[Dataset Info]')
    print(f'Train samples: {len(data_train)}')
    print(f'Valid samples: {len(data_valid)}')
    print(f'Test samples: {len(data_test)}\n')

    # Criar estrutura para histórico de métricas
    metrics_history = {"epochs": [], "config": {}}
    metrics_path = os.path.join(args.model_dir, args.save_model + '_metrics_history.json')

    # Salvar configuração do experimento
    metrics_history["config"] = {
        "model": args.save_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss_weights": args.loss_w,
        "image_size": args.crop_size,
        "n_classes": args.n_classes,
        "train_samples": len(data_train),
        "valid_samples": len(data_valid),
        "test_samples": len(data_test),
    }

    # Carregar histórico existente se houver
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_history = json.load(f)
        print(f'[INFO] Histórico de métricas carregado: {metrics_path}\n')

    # Train and val
    print('\n' + '='*80)
    print('STARTING TRAINING')
    print('='*80 + '\n')

    for epoch in range(0, args.epochs):
        train_loss = train_step(train_loader, model, epoch, optimizer, criterion, args)
        
        # ----------------------
        # VALIDATION WITH FULL METRICS
        # ----------------------
        model.eval()
        validation_loss = 0.0
        valid_all_preds = []
        valid_all_labels = []
        valid_all_probs = []
        
        with torch.no_grad():
            for imagesA, imagesB, imagesC, labels in valid_loader:
                imagesA = imagesA.to(device)
                imagesB = imagesB.to(device)
                imagesC = imagesC.to(device)
                labels = labels.to(device)
                
                # Forward pass
                _, _, _, _, outputs = model(imagesA, imagesB, imagesC)
                
                # Loss
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                
                # Predictions
                probs = outputs.cpu().numpy()
                preds = np.argmax(probs, axis=1)
                labels_np = np.argmax(labels.cpu().numpy(), axis=1)
                
                valid_all_preds.append(preds)
                valid_all_labels.append(labels_np)
                valid_all_probs.append(probs)
        
        validation_loss /= len(valid_loader)
        
        # Concatenar predições
        valid_all_preds = np.concatenate(valid_all_preds)
        valid_all_labels = np.concatenate(valid_all_labels)
        valid_all_probs = np.concatenate(valid_all_probs)
        
        # Calcular métricas de validação
        valid_metrics = compute_metric(valid_all_labels, valid_all_probs, target_names=["Good", "Usable", "Reject"])
        valid_acc = float((valid_all_preds == valid_all_labels).mean())
        valid_cm = confusion_matrix(valid_all_labels, valid_all_preds, labels=[0, 1, 2])
        
        print(f'\nEpoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {validation_loss:.4f}')
        print(f'Valid Acc: {valid_acc:.4f} | Valid F1: {np.mean(valid_metrics["F1"]):.4f} | Valid AUC: {valid_metrics["AUC"]:.4f}')
        print(f'Current Valid Loss: {validation_loss:.4f} | Best Valid Loss: {best_metric:.4f} at epoch: {best_iter+1}')
        
        # ----------------------
        # SAVE METRICS TO JSON
        # ----------------------
        epoch_metrics = {
            "epoch": epoch + 1,
            "train": {
                "loss": float(train_loss),
            },
            "valid": {
                "loss": float(validation_loss),
                "acc": float(valid_acc),
                "f1_macro": float(np.mean(valid_metrics["F1"])),
                "precision_macro": float(np.mean(valid_metrics["Precision"])),
                "sensitivity_macro": float(np.mean(valid_metrics["Sensitivity"])),
                "specificity_macro": float(np.mean(valid_metrics["Specificity"])),
                "auc": float(valid_metrics["AUC"]),
                "kappa": float(cohen_kappa_score(valid_all_labels, valid_all_preds)),
                "confusion_matrix": valid_cm.tolist(),
                "f1_per_class": {"Good": float(valid_metrics["F1"][0].item()), 
                                 "Usable": float(valid_metrics["F1"][1].item()), 
                                 "Reject": float(valid_metrics["F1"][2].item())},
                "sensitivity_per_class": {"Good": float(valid_metrics["Sensitivity"][0].item()), 
                                          "Usable": float(valid_metrics["Sensitivity"][1].item()), 
                                          "Reject": float(valid_metrics["Sensitivity"][2].item())},
                "specificity_per_class": {"Good": float(valid_metrics["Specificity"][0].item()), 
                                          "Usable": float(valid_metrics["Specificity"][1].item()), 
                                          "Reject": float(valid_metrics["Specificity"][2].item())},
            },
        }
        
        # Adicionar métricas da época ao histórico
        metrics_history["epochs"].append(epoch_metrics)
        
        # Salvar JSON atualizado
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)

        # save model
        if best_metric > validation_loss:
            best_metric = validation_loss
            best_iter = epoch
            model_save_file = os.path.join(args.save_dir, args.save_model + '.tar')
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            
            # Set model to train mode before saving
            model.train()
            torch.save({
                'state_dict': model.state_dict(), 
                'best_loss': best_metric,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'valid_f1': np.mean(valid_metrics["F1"]),
            }, model_save_file)
            print(f'✓ Model improved! Saved to {model_save_file}')
            print(f'  Valid Acc: {valid_acc:.4f} | Valid F1: {np.mean(valid_metrics["F1"]):.4f}\n')
        else:
            # Set back to train mode for next epoch
            model.train()
            print()

    print('\n' + '='*80)
    print(f'TRAINING COMPLETED | Best model at epoch {best_iter+1} with validation loss: {best_metric:.4f}')
    print('='*80 + '\n')

    # Load best model for testing
    print('[INFO] Loading best model for testing...')
    loaded_model = torch.load(os.path.join(args.save_dir, args.save_model + '.tar'))
    model.load_state_dict(loaded_model['state_dict'])
    model.to(device)
    print('[INFO] Best model loaded!\n')


    # Testing on test set
    print('='*80)
    print('EVALUATING ON TEST SET')
    print('='*80 + '\n')

    outPRED_mcs = torch.FloatTensor().cuda()
    test_labels_all = []
    model.eval()
    iters_per_epoch = len(test_loader)
    bar = Bar('Processing {}'.format('test inference'), max=len(test_loader))
    bar.check_tty = False
    for epochID, (imagesA, imagesB, imagesC, labels) in enumerate(test_loader):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        begin_time = time.time()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
        outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
        
        # Collect labels
        labels_np = np.argmax(labels.cpu().numpy(), axis=1)
        test_labels_all.append(labels_np)
        
        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                               batch_time=batch_time * (iters_per_epoch - epochID) / 60)
        bar.next()
    bar.finish()
    
    # Concatenate all test labels
    GT_QA_list = np.concatenate(test_labels_all)

    print('\n[INFO] Saving test predictions...')
    # save result into excel:
    save_output(label_test_file, outPRED_mcs, args, save_file=save_file_name)
    print(f'[INFO] Predictions saved to {save_file_name}\n')

    # evaluation:
    df_gt = pd.read_csv(label_test_file)
    img_list = df_gt["image"].tolist()
    img_num = len(img_list)
    label_list = ["Good", "Usable", "Reject"]

    df_tmp = pd.read_csv(save_file_name)
    predict_tmp = np.zeros([img_num, 3])
    for idx in range(3):
        predict_tmp[:, idx] = np.array(df_tmp[label_list[idx]].tolist())

    print('[INFO] Computing metrics...\n')
    tmp_report = compute_metric(GT_QA_list, predict_tmp, target_names=label_list)

    # Quick summary
    print('='*80)
    print('TEST SET RESULTS (QUICK SUMMARY)')
    print('='*80)
    print(f' Accuracy:    {np.mean(tmp_report["Accuracy"]):.4f}')
    print(f' Precision:   {np.mean(tmp_report["Precision"]):.4f}')
    print(f' Sensitivity: {np.mean(tmp_report["Sensitivity"]):.4f}')
    print(f' F1-Score:    {np.mean(tmp_report["F1"]):.4f}')
    print(f' AUC:         {tmp_report["AUC"]:.4f}')
    print('='*80 + '\n')

    # Generate detailed report
    print('[INFO] Generating detailed report...')
    report = generate_detailed_report(GT_QA_list, predict_tmp, tmp_report, args, best_iter, best_metric, label_list)

    # Save report
    report_file = os.path.join(args.model_dir, args.save_model + '_test_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[INFO] Detailed report saved to: {report_file}')
    print('\n' + '='*80)
    print('EVALUATION COMPLETED')
    print('='*80)