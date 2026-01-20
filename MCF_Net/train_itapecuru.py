"""
Script auxiliar para treinar MCF-Net com dataset Itapecuru
Simplifica a execução do Main_EyeQuality.py com parâmetros predefinidos
"""

import subprocess
import sys

def train_mcfnet(
    epochs=20,
    batch_size=4,
    lr=0.01,
    save_model_name='DenseNet121_Itapecuru'
):
    """
    Executa o treinamento da MCF-Net
    
    Args:
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        lr: Learning rate
        save_model_name: Nome do modelo a ser salvo
    """
    
    cmd = [
        sys.executable,  # Python executable
        'Main_EyeQuality.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--save_model', save_model_name
    ]
    
    print("=" * 80)
    print("MCF-NET TRAINING - ITAPECURU DATASET")
    print("=" * 80)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Model Name: {save_model_name}")
    print("=" * 80)
    print()
    
    # Execute
    subprocess.run(cmd)


if __name__ == '__main__':
    # Configurações padrão
    train_mcfnet(
        epochs=20,
        batch_size=4,
        lr=0.01,
        save_model_name='DenseNet121_Itapecuru'
    )
