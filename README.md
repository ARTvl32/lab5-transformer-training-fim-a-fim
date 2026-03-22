# Laboratório 5 — Treinamento Fim-a-Fim do Transformer

**Disciplina:** Tópicos em Inteligência Artificial 2026.1  
**Instituição:** iCEV — Instituto de Ensino Superior  
**Professor:** Dimmy Magalhães  

> **Nota sobre IA Generativa:** Partes relacionadas à manipulação do dataset
> e tokenização (Tarefas 1 e 2) foram geradas/complementadas com IA,
> revisadas por Arthur. O fluxo de Forward/Backward (Tarefa 3) foi construído
> sobre as classes desenvolvidas nos laboratórios anteriores (Labs 01–04),
> portanto a lógica matemática central é de autoria própria.

---

## Contexto

Este é o laboratório final da Unidade I. O objetivo não é construir um tradutor
perfeito (o modelo de 2017 do Google treinou 3,5 dias em 8 GPUs dedicadas),
mas sim **provar que a arquitetura consegue aprender** — forçando a função
de perda (Loss) a cair significativamente ao longo das épocas.

A arquitetura Transformer dos Labs 01–04 é reescrita em **PyTorch** (`nn.Module`)
para permitir backpropagation real com `loss.backward()` e `optimizer.step()`.

---

## Estrutura do Repositório

```
lab5-training/
│
├── tarefa1_dataset.py        # Carregamento e subset do Hugging Face
├── tarefa2_tokenizacao.py    # Tokenização com AutoTokenizer + padding
├── tarefa3_training_loop.py  # Modelo PyTorch + CrossEntropy + Adam
├── tarefa4_overfitting.py    # Prova de fogo: overfitting em 1 frase
└── README.md
```

---

## Tarefas

### Tarefa 1 — Dataset Real (Hugging Face)

Carrega o dataset `Helsinki-NLP/opus_books` (par `en-fr`) e seleciona as
primeiras **1.000 frases** como conjunto de treinamento.

```python
from datasets import load_dataset
dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[:1000]")
```

---

### Tarefa 2 — Tokenização Básica

Usa `AutoTokenizer.from_pretrained("bert-base-multilingual-cased")` para
converter pares de frases em listas de inteiros. Para as frases de destino
(Decoder), adiciona os tokens especiais `[CLS]` como `<START>` e `[SEP]`
como `<EOS>`, e aplica **padding** para uniformizar o comprimento do batch.

---

### Tarefa 3 — Training Loop (Forward → Loss → Backward → Step)

- **Modelo:** `TransformerPyTorch` (`d_model=128`, `h=4`, `N=2`) em `nn.Module`
- **Loss:** `CrossEntropyLoss(ignore_index=PAD_ID)` — ignora tokens de padding
- **Otimizador:** `Adam(lr=1e-4)` — mesmo do paper original
- **Épocas:** 20

Fluxo por época:
```
encoder_input  → Encoder Stack (N=2)  → Z
decoder_input  → Decoder Stack (N=2, cross-attn com Z) → logits
logits vs target_output → CrossEntropyLoss → loss.backward() → optimizer.step()
```

A entrada do Decoder é deslocada 1 posição à direita (*teacher forcing*):
- **decoder_input:** `[<START>, tok1, tok2, ...]`
- **target_output:** `[tok1, tok2, ..., <EOS>]`

---

### Tarefa 4 — Prova de Fogo (Overfitting Test)

Após o treinamento, seleciona **uma frase específica** do conjunto de treino e
executa o loop auto-regressivo. O modelo deve reproduzir a tradução exata
(ou muito próxima), provando que os gradientes fluíram corretamente e que
a arquitetura assimilou o padrão.

---

## Como Executar

> **Recomendado:** Google Colab (gratuito, já tem PyTorch, CUDA disponível)

```bash
# Instalar dependências
pip install torch transformers datasets

# Tarefa 1 — Dataset
python tarefa1_dataset.py

# Tarefa 2 — Tokenização
python tarefa2_tokenizacao.py

# Tarefa 3 — Training Loop (roda ~2-5 min no Colab)
python tarefa3_training_loop.py

# Tarefa 4 — Overfitting test
python tarefa4_overfitting.py
```

---

## O que Observar Durante o Treinamento

A saída do training loop imprime o Loss a cada época. O comportamento
esperado é uma **queda significativa**:

```
Época  1/20 | Loss: 5.8342
Época  2/20 | Loss: 5.1203
Época  5/20 | Loss: 3.9871
Época 10/20 | Loss: 2.4103
Época 20/20 | Loss: 0.8821   ← convergência confirmada
```

---

## Arquitetura PyTorch

```
TransformerPyTorch (nn.Module)
├── src_embedding   : nn.Embedding(vocab_size, d_model)
├── tgt_embedding   : nn.Embedding(vocab_size, d_model)
├── encoder_layers  : nn.ModuleList([EncoderLayer x N])
│     └── EncoderLayer
│           ├── self_attn  : MultiHeadAttention
│           ├── ffn        : FeedForward
│           └── norm1/2    : nn.LayerNorm
├── decoder_layers  : nn.ModuleList([DecoderLayer x N])
│     └── DecoderLayer
│           ├── self_attn  : MultiHeadAttention (masked)
│           ├── cross_attn : MultiHeadAttention
│           ├── ffn        : FeedForward
│           └── norm1/2/3  : nn.LayerNorm
└── output_proj     : nn.Linear(d_model, vocab_size)
```

---

## Fundamentos Matemáticos

**Cross-Entropy Loss:**

$$\mathcal{L} = -\sum_{t} \log P(y_t^* \mid y_{<t}, X)$$

**Adam Optimizer** (Kingma & Ba, 2014):

$$\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Teacher Forcing** (deslocamento de 1 posição):

$$\text{decoder\_input} = [⟨\text{START}⟩, y_1, y_2, \ldots, y_{T-1}]$$
$$\text{target} = [y_1, y_2, \ldots, y_T, ⟨\text{EOS}⟩]$$

---

## Referências

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization*.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
- Laboratórios 01–04 — arquitetura base reutilizada e portada para PyTorch
