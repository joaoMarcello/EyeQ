# MCF-Net - Adaptado para Dataset Itapecuru

## Resumo das Adaptações

O código MCF-Net foi adaptado para funcionar com o dataset Itapecuru, incluindo:

1. **Conjunto de dados separado para teste**: Além de treino e validação, agora há um conjunto de teste independente
2. **Loop de treinamento funcional**: O treinamento usa o conjunto de validação para selecionar o melhor modelo
3. **Relatório detalhado de métricas**: Geração automática de relatório completo ao final do treinamento

---

## Estrutura dos Dados

O dataset Itapecuru está organizado da seguinte forma:

```
C:\Users\Public\Documents\DATASETS\Itapecuru\ORIGINAL_CLEAN_768x768\
├── [todas as imagens em um único diretório]
```

Os CSVs de divisão estão em:
```
data/revised/color/
├── itapecuru_train_color_rev.csv   (treino)
├── itapecuru_valid_color_rev.csv   (validação - seleção de modelo)
└── itapecuru_test_color_rev.csv    (teste - avaliação final)
```

**Importante**: Todas as imagens (treino, validação e teste) ficam no mesmo diretório. A separação é feita apenas pelos CSVs.

---

## Principais Mudanças no Código

### 1. Main_EyeQuality.py

#### Adições:
- Importação de `datetime`, `confusion_matrix`, `cohen_kappa_score`, `roc_auc_score`
- Função `generate_detailed_report()` para gerar relatórios detalhados
- Argumento `--save_dir` para diretório de checkpoints
- Dataloader para conjunto de teste (`test_loader`)
- Contagem de samples por conjunto (train/valid/test)

#### Modificações:
- **Treinamento**: Loop de treinamento agora está ativo (antes estava comentado)
- **Validação**: Conjunto de validação é usado para selecionar o melhor modelo durante o treino
- **Teste**: Conjunto de teste é usado apenas para avaliação final após o treinamento
- **Salvamento**: Melhor modelo é salvo em `./checkpoints/` baseado na menor loss de validação
- **Relatório**: Após o treinamento, gera um relatório completo em formato texto

### 2. EyeQ_loader.py

#### Modificações:
- Função `load_eyeQ_excel()` agora suporta múltiplos formatos de imagem
- Detecta automaticamente se a imagem tem extensão `.jpg`, `.jpeg` ou `.png`
- Mantém compatibilidade com dataset EyeQ original

---

## Como Usar

### Opção 1: Usar o script auxiliar (Recomendado)

```bash
cd MCF_Net
python train_itapecuru.py
```

### Opção 2: Executar diretamente

```bash
cd MCF_Net
python Main_EyeQuality.py --epochs 20 --batch-size 4 --lr 0.01 --save_model DenseNet121_Itapecuru
```

### Parâmetros Disponíveis

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 20 | Número de épocas de treinamento |
| `--batch-size` | 4 | Tamanho do batch |
| `--lr` | 0.01 | Learning rate |
| `--save_model` | DenseNet121_Itapecuru | Nome do modelo a salvar |
| `--model_dir` | ./result/ | Diretório para salvar resultados |
| `--save_dir` | ./checkpoints/ | Diretório para salvar checkpoints |
| `--pre_model` | None | Nome do modelo pré-treinado (se houver) |

---

## Processo de Treinamento

1. **Carregamento dos dados**: Carrega treino, validação e teste
2. **Loop de treinamento**:
   - Para cada época:
     - Treina no conjunto de treino
     - Avalia no conjunto de validação
     - Se a loss de validação melhorar, salva o modelo
3. **Carregamento do melhor modelo**: Carrega o modelo com melhor performance na validação
4. **Avaliação no conjunto de teste**: Avalia o melhor modelo no conjunto de teste
5. **Geração de relatório**: Cria relatório detalhado com todas as métricas

---

## Arquivos Gerados

Após o treinamento, os seguintes arquivos são gerados:

### 1. Checkpoint do Modelo
- **Localização**: `./checkpoints/DenseNet121_Itapecuru.tar`
- **Conteúdo**:
  - `state_dict`: Pesos do modelo
  - `best_loss`: Melhor loss de validação
  - `epoch`: Época do melhor modelo
  - `optimizer`: Estado do otimizador
  - `valid_acc`: Acurácia de validação
  - `valid_f1`: F1-score de validação

### 2. Histórico de Métricas (JSON) ⭐ NOVO
- **Localização**: `./result/DenseNet121_Itapecuru_metrics_history.json`
- **Conteúdo por época**:
  - **Train**: loss
  - **Validation**: loss, accuracy, F1-score, precision, sensitivity, specificity, AUC, Cohen's Kappa, confusion matrix
  - Métricas por classe (F1, sensitivity, specificity)
- **Uso**: Permite plotar gráficos de evolução do treinamento

### 3. Predições no Conjunto de Teste
- **Localização**: `./result/DenseNet121_Itapecuru_test_predictions.csv`
- **Formato**:
  ```csv
  image_name,Good,Usable,Reject
  paciente1/imagem1.jpg,0.95,0.04,0.01
  paciente2/imagem2.jpg,0.10,0.85,0.05
  ...
  ```

### 4. Relatório Detalhado
- **Localização**: `./result/DenseNet121_Itapecuru_test_report.txt`
- **Conteúdo**:
  - Configuração do experimento
  - Métricas por classe (Precision, Recall, Specificity, F1, IoU)
  - Métricas agregadas (Macro e Weighted)
  - Matriz de confusão
  - Cohen's Kappa
  - AUC

---

## Visualização de Métricas

Após o treinamento, você pode gerar gráficos das métricas usando o script de plotagem:

```bash
python plot_training_metrics.py --metrics_file result/DenseNet121_Itapecuru_metrics_history.json
```

Isso gerará os seguintes gráficos em `./result/plots/`:

1. **loss_curves.png**: Curvas de loss (treino e validação)
2. **accuracy_curve.png**: Acurácia no conjunto de validação
3. **f1_scores.png**: F1-scores (macro e por classe)
4. **sensitivity_specificity.png**: Sensitivity e Specificity macro
5. **all_metrics.png**: Todas as métricas principais em um único gráfico
6. **confusion_matrix_last_epoch.png**: Matriz de confusão da última época
7. **per_class_metrics.png**: Sensitivity e Specificity por classe

### Exemplo de Uso:

```bash
# Após o treinamento
cd MCF_Net
python plot_training_metrics.py --metrics_file result/DenseNet121_Itapecuru_metrics_history.json

# Ou especificar diretório de saída customizado
python plot_training_metrics.py --metrics_file result/DenseNet121_Itapecuru_metrics_history.json --output_dir meus_graficos
```

---

## Exemplo de Saída do Relatório

```
================================================================================
RELATÓRIO DE AVALIAÇÃO - MCF-NET (ITAPECURU DATASET)
================================================================================

Data/Hora: 2026-01-20 15:30:45
Modelo: DenseNet121_Itapecuru
Melhor época: 15/20
Melhor validation loss: 0.3245

================================================================================
MÉTRICAS POR CLASSE - TEST SET
================================================================================
Classe       |    Suporte |   Precision |      Recall | Specificity |   F1-Score | IoU/Jaccard
--------------------------------------------------------------------------------------------------------------------
Good         |        150 |      0.8500 |      0.9000 |      0.9200 |     0.8700 |      0.7800
Usable       |        200 |      0.7800 |      0.8200 |      0.8900 |     0.8000 |      0.6900
Reject       |        100 |      0.9200 |      0.8800 |      0.9500 |     0.9000 |      0.8200

================================================================================
MÉTRICAS AGREGADAS - TEST SET
================================================================================
Métrica                   |   Macro (avg) | Weighted (avg)
----------------------------------------------------------------------
Precision                 |        0.8500 |         0.8300
Recall (Sensitivity)      |        0.8667 |         0.8500
Specificity               |        0.9200 |         0.9100
F1-Score                  |        0.8567 |         0.8400
IoU/Jaccard               |        0.7633 |         0.7400
Accuracy                  |        0.8667 |         0.8667
Cohen Kappa               |        0.7850 |         0.7850
AUC                       |        0.9100 |         0.9100

================================================================================
MATRIZ DE CONFUSÃO - TEST SET
================================================================================
            Predito
                    Good     Usable     Reject
Real
Good                 135         10          5
Usable                15        164         21
Reject                 5          7         88
```

---

## Métricas Calculadas

O relatório inclui as seguintes métricas:

### Por Classe
- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **IoU (Jaccard)**: TP / (TP + FP + FN)
- **Suporte**: Número de amostras reais de cada classe

### Agregadas
- **Macro Average**: Média simples das métricas por classe
- **Weighted Average**: Média ponderada pelo suporte de cada classe

### Globais
- **Accuracy**: Taxa de acertos geral
- **Cohen's Kappa**: Concordância corrigida pelo acaso
- **AUC**: Área sob a curva ROC (média das classes)

---

## Diferenças em Relação ao Código Original

| Aspecto | Original | Adaptado |
|---------|----------|----------|
| Dataset | EyeQ | Itapecuru |
| Formato de imagem | .png (convertido de .jpeg) | .jpg (nativo) |
| Organização | Pastas separadas train/test | Todas as imagens na mesma pasta |
| Loop de treino | Comentado | Ativo e funcional |
| Validação | Não usada para seleção | Usada para selecionar melhor modelo |
| Teste | Usado para inferência direta | Usado após carregar melhor modelo |
| Relatório | Print simples | Relatório detalhado em arquivo .txt |
| Checkpoints | Não salvos | Salvos em ./checkpoints/ |

---

## Notas Importantes

1. **GPU**: O código está configurado para usar GPU (CUDA). Certifique-se de ter PyTorch com suporte CUDA instalado.

2. **Memória**: Com batch_size=4, o treinamento requer aproximadamente 6-8GB de memória GPU. Ajuste o batch_size se necessário.

3. **Tempo de treinamento**: Com 20 épocas, o treinamento pode levar várias horas dependendo do hardware.

4. **Validação vs Teste**:
   - **Validação**: Usada DURANTE o treinamento para selecionar o melhor modelo
   - **Teste**: Usada APÓS o treinamento para avaliação final (nunca visto durante o treino)

5. **Reproducibilidade**: O seed está fixado em 0 para garantir resultados reproduzíveis.

---

## Troubleshooting

### Erro: "CUDA out of memory"
**Solução**: Reduza o `--batch-size` para 2 ou 1

### Erro: "FileNotFoundError" nos CSVs
**Solução**: Verifique se os caminhos dos CSVs estão corretos em relação ao diretório MCF_Net/

### Erro: "Image not found"
**Solução**: Verifique se o caminho em `data_root` está correto e se as imagens existem no diretório

---

## Contato

Para dúvidas ou problemas, entre em contato com o desenvolvedor do projeto.
