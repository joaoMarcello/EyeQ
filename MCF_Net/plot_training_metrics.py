"""
Script para plotar métricas de treinamento da MCF-Net a partir do JSON gerado
Uso: python plot_training_metrics.py --metrics_file result/DenseNet121_Itapecuru_metrics_history.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_metrics(metrics_file):
    """Carrega métricas do arquivo JSON"""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_loss(epochs_data, save_dir):
    """Plota curvas de loss (treino e validação)"""
    epochs = [e['epoch'] for e in epochs_data]
    train_loss = [e['train']['loss'] for e in epochs_data]
    valid_loss = [e['valid']['loss'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, valid_loss, 's-', label='Valid Loss', linewidth=2, markersize=6)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Curvas de Loss - Treino vs Validação', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'loss_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_accuracy(epochs_data, save_dir):
    """Plota curva de acurácia de validação"""
    epochs = [e['epoch'] for e in epochs_data]
    valid_acc = [e['valid']['acc'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, valid_acc, 'o-', label='Valid Accuracy', color='green', linewidth=2, markersize=6)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Acurácia no Conjunto de Validação', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'accuracy_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_f1_scores(epochs_data, save_dir):
    """Plota F1-scores (macro e por classe)"""
    epochs = [e['epoch'] for e in epochs_data]
    f1_macro = [e['valid']['f1_macro'] for e in epochs_data]
    f1_good = [e['valid']['f1_per_class']['Good'] for e in epochs_data]
    f1_usable = [e['valid']['f1_per_class']['Usable'] for e in epochs_data]
    f1_reject = [e['valid']['f1_per_class']['Reject'] for e in epochs_data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, f1_macro, 'o-', label='F1 Macro', linewidth=2.5, markersize=7, color='black')
    plt.plot(epochs, f1_good, 's-', label='F1 Good', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(epochs, f1_usable, '^-', label='F1 Usable', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(epochs, f1_reject, 'd-', label='F1 Reject', linewidth=2, markersize=5, alpha=0.7)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Scores - Macro e Por Classe', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'f1_scores.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_sensitivity_specificity(epochs_data, save_dir):
    """Plota Sensitivity e Specificity (macro)"""
    epochs = [e['epoch'] for e in epochs_data]
    sensitivity = [e['valid']['sensitivity_macro'] for e in epochs_data]
    specificity = [e['valid']['specificity_macro'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sensitivity, 'o-', label='Sensitivity (Recall)', linewidth=2, markersize=6, color='blue')
    plt.plot(epochs, specificity, 's-', label='Specificity', linewidth=2, markersize=6, color='red')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Métrica', fontsize=12)
    plt.title('Sensitivity e Specificity - Validação', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'sensitivity_specificity.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_all_metrics(epochs_data, save_dir):
    """Plota todas as métricas principais em um único gráfico"""
    epochs = [e['epoch'] for e in epochs_data]
    acc = [e['valid']['acc'] for e in epochs_data]
    f1 = [e['valid']['f1_macro'] for e in epochs_data]
    precision = [e['valid']['precision_macro'] for e in epochs_data]
    sensitivity = [e['valid']['sensitivity_macro'] for e in epochs_data]
    specificity = [e['valid']['specificity_macro'] for e in epochs_data]
    
    plt.figure(figsize=(14, 7))
    plt.plot(epochs, acc, 'o-', label='Accuracy', linewidth=2, markersize=5)
    plt.plot(epochs, f1, 's-', label='F1-Score', linewidth=2, markersize=5)
    plt.plot(epochs, precision, '^-', label='Precision', linewidth=2, markersize=5)
    plt.plot(epochs, sensitivity, 'd-', label='Sensitivity', linewidth=2, markersize=5)
    plt.plot(epochs, specificity, 'v-', label='Specificity', linewidth=2, markersize=5)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Métrica', fontsize=12)
    plt.title('Todas as Métricas de Validação', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'all_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_confusion_matrices(epochs_data, save_dir):
    """Plota matriz de confusão da última época"""
    last_epoch = epochs_data[-1]
    cm = np.array(last_epoch['valid']['confusion_matrix'])
    epoch_num = last_epoch['epoch']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Usable', 'Reject'],
                yticklabels=['Good', 'Usable', 'Reject'],
                cbar_kws={'label': 'Contagem'})
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title(f'Matriz de Confusão - Validação (Época {epoch_num})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'confusion_matrix_last_epoch.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def plot_per_class_metrics(epochs_data, save_dir):
    """Plota métricas por classe em subplots"""
    epochs = [e['epoch'] for e in epochs_data]
    
    # Sensitivity por classe
    sens_good = [e['valid']['sensitivity_per_class']['Good'] for e in epochs_data]
    sens_usable = [e['valid']['sensitivity_per_class']['Usable'] for e in epochs_data]
    sens_reject = [e['valid']['sensitivity_per_class']['Reject'] for e in epochs_data]
    
    # Specificity por classe
    spec_good = [e['valid']['specificity_per_class']['Good'] for e in epochs_data]
    spec_usable = [e['valid']['specificity_per_class']['Usable'] for e in epochs_data]
    spec_reject = [e['valid']['specificity_per_class']['Reject'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sensitivity
    axes[0].plot(epochs, sens_good, 'o-', label='Good', linewidth=2, markersize=5)
    axes[0].plot(epochs, sens_usable, 's-', label='Usable', linewidth=2, markersize=5)
    axes[0].plot(epochs, sens_reject, '^-', label='Reject', linewidth=2, markersize=5)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Sensitivity (Recall)', fontsize=12)
    axes[0].set_title('Sensitivity por Classe', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Specificity
    axes[1].plot(epochs, spec_good, 'o-', label='Good', linewidth=2, markersize=5)
    axes[1].plot(epochs, spec_usable, 's-', label='Usable', linewidth=2, markersize=5)
    axes[1].plot(epochs, spec_reject, '^-', label='Reject', linewidth=2, markersize=5)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Specificity', fontsize=12)
    axes[1].set_title('Specificity por Classe', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'per_class_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVED] {save_path}')
    plt.close()

def print_summary(metrics_history):
    """Imprime resumo das métricas"""
    config = metrics_history['config']
    epochs_data = metrics_history['epochs']
    
    print("\n" + "="*80)
    print("RESUMO DO TREINAMENTO")
    print("="*80)
    print(f"Modelo: {config['model']}")
    print(f"Épocas treinadas: {len(epochs_data)}/{config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Image size: {config['image_size']}")
    print(f"Train samples: {config['train_samples']}")
    print(f"Valid samples: {config['valid_samples']}")
    print(f"Test samples: {config['test_samples']}")
    
    # Melhor época
    best_epoch_idx = min(range(len(epochs_data)), key=lambda i: epochs_data[i]['valid']['loss'])
    best_epoch = epochs_data[best_epoch_idx]
    
    print("\n" + "="*80)
    print("MELHOR ÉPOCA (baseado em validation loss)")
    print("="*80)
    print(f"Época: {best_epoch['epoch']}")
    print(f"Train Loss: {best_epoch['train']['loss']:.4f}")
    print(f"Valid Loss: {best_epoch['valid']['loss']:.4f}")
    print(f"Valid Accuracy: {best_epoch['valid']['acc']:.4f}")
    print(f"Valid F1 (macro): {best_epoch['valid']['f1_macro']:.4f}")
    print(f"Valid Precision (macro): {best_epoch['valid']['precision_macro']:.4f}")
    print(f"Valid Sensitivity (macro): {best_epoch['valid']['sensitivity_macro']:.4f}")
    print(f"Valid Specificity (macro): {best_epoch['valid']['specificity_macro']:.4f}")
    print(f"Valid AUC: {best_epoch['valid']['auc']:.4f}")
    print(f"Valid Kappa: {best_epoch['valid']['kappa']:.4f}")
    
    print("\nF1-Scores por classe:")
    print(f"  Good:   {best_epoch['valid']['f1_per_class']['Good']:.4f}")
    print(f"  Usable: {best_epoch['valid']['f1_per_class']['Usable']:.4f}")
    print(f"  Reject: {best_epoch['valid']['f1_per_class']['Reject']:.4f}")
    
    # Última época
    last_epoch = epochs_data[-1]
    print("\n" + "="*80)
    print("ÚLTIMA ÉPOCA")
    print("="*80)
    print(f"Época: {last_epoch['epoch']}")
    print(f"Train Loss: {last_epoch['train']['loss']:.4f}")
    print(f"Valid Loss: {last_epoch['valid']['loss']:.4f}")
    print(f"Valid Accuracy: {last_epoch['valid']['acc']:.4f}")
    print(f"Valid F1 (macro): {last_epoch['valid']['f1_macro']:.4f}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Plota métricas de treinamento da MCF-Net')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Caminho para o arquivo JSON de métricas')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Diretório para salvar gráficos (default: mesmo dir do JSON)')
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not Path(args.metrics_file).exists():
        print(f"[ERRO] Arquivo não encontrado: {args.metrics_file}")
        return
    
    # Carregar métricas
    print(f"[INFO] Carregando métricas de: {args.metrics_file}")
    metrics_history = load_metrics(args.metrics_file)
    epochs_data = metrics_history['epochs']
    
    if len(epochs_data) == 0:
        print("[ERRO] Nenhuma época encontrada no arquivo de métricas!")
        return
    
    # Definir diretório de saída
    if args.output_dir is None:
        output_dir = Path(args.metrics_file).parent / 'plots'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[INFO] Salvando gráficos em: {output_dir}\n")
    
    # Imprimir resumo
    print_summary(metrics_history)
    
    # Gerar gráficos
    print("[INFO] Gerando gráficos...")
    plot_loss(epochs_data, output_dir)
    plot_accuracy(epochs_data, output_dir)
    plot_f1_scores(epochs_data, output_dir)
    plot_sensitivity_specificity(epochs_data, output_dir)
    plot_all_metrics(epochs_data, output_dir)
    plot_confusion_matrices(epochs_data, output_dir)
    plot_per_class_metrics(epochs_data, output_dir)
    
    print(f"\n[INFO] Todos os gráficos foram salvos em: {output_dir}")
    print("[INFO] Arquivos gerados:")
    print("  - loss_curves.png")
    print("  - accuracy_curve.png")
    print("  - f1_scores.png")
    print("  - sensitivity_specificity.png")
    print("  - all_metrics.png")
    print("  - confusion_matrix_last_epoch.png")
    print("  - per_class_metrics.png")

if __name__ == '__main__':
    main()
