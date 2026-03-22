"""
Laboratório 5 — Tarefa 3: O Motor de Otimização (Training Loop)
================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Reimplementa a arquitetura Transformer dos Labs 01-04 em PyTorch (nn.Module)
para habilitar backpropagation real com loss.backward() e optimizer.step().

A lógica matemática dos blocos (Scaled Dot-Product Attention, Add & Norm,
FFN, Máscara Causal, Cross-Attention) é a mesma desenvolvida nos laboratórios
anteriores — apenas portada do NumPy para tensores PyTorch com gradientes.

Fluxo de treinamento por época
-------------------------------
    Para cada batch:
        1. Forward pass:
           a. src_ids  → Embedding + PE → Encoder(N camadas) → Z
           b. dec_ids  → Embedding + PE → Decoder(N camadas, cross-attn Z) → logits
        2. Loss = CrossEntropyLoss(logits, target_ids, ignore_index=PAD)
        3. Backward: loss.backward()
        4. Step:     optimizer.step()
        5. optimizer.zero_grad()

Hiperparâmetros (viáveis para CPU/Colab gratuito)
-------------------------------------------------
    d_model    = 128
    n_heads    = 4
    N          = 2    (camadas do Encoder e Decoder)
    d_ff       = 512
    max_len    = 64
    batch_size = 32
    epochs     = 20
    lr         = 1e-4

Dependências
------------
    pip install torch transformers datasets
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tarefa1_dataset    import load_translation_subset
from tarefa2_tokenizacao import load_tokenizer, build_batches, PAD_ID, START_ID, EOS_ID


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------

D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 512
MAX_LEN    = 64
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Codificação posicional sinusoidal (Vaswani et al., 2017).
    Somada ao embedding antes de entrar no Encoder/Decoder.
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        PE  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        PE[:, 0::2] = torch.sin(pos * div)
        PE[:, 1::2] = torch.cos(pos * div)
        PE = PE.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("PE", PE)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.dropout(x + self.PE[:, :x.size(1)])


# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention (com suporte a máscara)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax( QK^T / sqrt(d_k) + mask ) * V

    Q, K : (..., T, d_k)
    V    : (..., T, d_v)
    mask : (..., T, T) com -1e9 nas posições bloqueadas
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention com h cabeças paralelas.
    Cada cabeça opera em d_k = d_model / h dimensões.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h    = n_heads
        self.d_k  = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        # Projeção + split em h cabeças: (B, T, d_model) -> (B, h, T, d_k)
        Q = self.Wq(query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.Wk(key  ).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.Wv(value).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # Atenção por cabeça
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concat + projeção final: (B, h, T, d_k) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.Wo(attn_out)


# ---------------------------------------------------------------------------
# 4. Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """FFN(x) = max(0, xW1+b1)W2+b2  com expansão d_model -> d_ff -> d_model."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Uma camada do Encoder:
        x → Self-Attention → Add&Norm → FFN → Add&Norm → Z
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sub-camada 1: Self-Attention + Add & Norm
        attn = self.self_attn(x, x, x, src_mask)
        x    = self.norm1(x + self.dropout(attn))
        # Sub-camada 2: FFN + Add & Norm
        x    = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# 6. Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Uma camada do Decoder:
        y → Masked Self-Attention → Add&Norm
          → Cross-Attention(Z)   → Add&Norm
          → FFN                  → Add&Norm → saída
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        # Sub-camada 1: Masked Self-Attention + Add & Norm
        attn1 = self.self_attn(y, y, y, tgt_mask)
        y     = self.norm1(y + self.dropout(attn1))
        # Sub-camada 2: Cross-Attention (Q←y, K/V←Z) + Add & Norm
        attn2 = self.cross_attn(y, Z, Z, src_mask)
        y     = self.norm2(y + self.dropout(attn2))
        # Sub-camada 3: FFN + Add & Norm
        y     = self.norm3(y + self.dropout(self.ffn(y)))
        return y


# ---------------------------------------------------------------------------
# 7. Transformer completo (nn.Module)
# ---------------------------------------------------------------------------

class TransformerPyTorch(nn.Module):
    """
    Arquitetura Encoder-Decoder completa portada dos Labs 01-04 para PyTorch.

    Parâmetros
    ----------
    vocab_size : int   — tamanho do vocabulário
    d_model    : int   — dimensão dos embeddings (128)
    n_heads    : int   — número de cabeças de atenção (4)
    n_layers   : int   — número de camadas Encoder e Decoder (2)
    d_ff       : int   — dimensão interna do FFN (512)
    max_len    : int   — comprimento máximo para Positional Encoding
    dropout    : float — taxa de dropout
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings (compartilha vocabulário entre Encoder e Decoder)
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Pilha do Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Pilha do Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Projeção final: d_model → vocab_size
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Inicialização de Xavier para pesos lineares
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, size, device):
        """Máscara triangular superior com -1e9 (bloqueia o futuro)."""
        mask = torch.triu(torch.full((size, size), -1e9, device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T) para broadcast

    def encode(self, src):
        """
        Processa a frase de entrada pelo Encoder.
        src : (B, T_src)  — IDs dos tokens
        Retorna Z : (B, T_src, d_model)
        """
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, tgt, Z):
        """
        Processa os tokens já gerados pelo Decoder.
        tgt : (B, T_tgt)           — IDs dos tokens gerados
        Z   : (B, T_src, d_model)  — memória do Encoder
        Retorna logits : (B, T_tgt, vocab_size)
        """
        T_tgt   = tgt.size(1)
        tgt_mask = self._causal_mask(T_tgt, tgt.device)
        y = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            y = layer(y, Z, tgt_mask=tgt_mask)
        return self.output_proj(y)   # (B, T_tgt, vocab_size)

    def forward(self, src, tgt):
        """Forward pass completo: Encoder → Z → Decoder → logits."""
        Z      = self.encode(src)
        logits = self.decode(tgt, Z)
        return logits


# ---------------------------------------------------------------------------
# 8. Training Loop
# ---------------------------------------------------------------------------

def train(model, batches, epochs=EPOCHS, lr=LR, device=DEVICE):
    """
    Executa o loop de treinamento completo.

    Para cada batch e cada época:
        Forward → CrossEntropyLoss → backward() → optimizer.step()

    A CrossEntropyLoss usa ignore_index=PAD_ID para que tokens de
    padding não contribuam para o gradiente.

    Parâmetros
    ----------
    model   : TransformerPyTorch
    batches : list[dict]  — saída de build_batches()
    epochs  : int
    lr      : float
    device  : torch.device

    Retorna
    -------
    loss_history : list[float]  — loss médio por época
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    loss_history = []

    print(f"\nDispositivo: {device}")
    print(f"Épocas: {epochs}  |  Batches por época: {len(batches)}")
    print(f"d_model={D_MODEL}, n_heads={N_HEADS}, N={N_LAYERS}, d_ff={D_FF}\n")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        total_loss   = 0.0
        total_tokens = 0

        for batch in batches:
            src     = batch["src"].to(device)       # (B, T_src)
            dec_inp = batch["dec_inp"].to(device)   # (B, T_tgt)
            tgt     = batch["tgt"].to(device)       # (B, T_tgt)

            # --- Forward ---
            logits = model(src, dec_inp)             # (B, T_tgt, vocab_size)

            # Reshape para CrossEntropyLoss: (B*T, V) vs (B*T,)
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                tgt.view(B * T),
            )

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (previne exploding gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Acumular loss ponderado pelos tokens reais (não-padding)
            n_tokens    = (tgt != PAD_ID).sum().item()
            total_loss  += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        loss_history.append(avg_loss)

        print(f"Época {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    print("-" * 50)
    queda = loss_history[0] - loss_history[-1]
    print(f"\n✓ Loss inicial : {loss_history[0]:.4f}")
    print(f"✓ Loss final   : {loss_history[-1]:.4f}")
    print(f"✓ Queda total  : {queda:.4f}  ({queda/loss_history[0]*100:.1f}%)")

    return loss_history


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 3 — Training Loop (Forward → Loss → Backward → Step)")
    print("=" * 60)

    # Carregar dados e tokenizador
    pairs     = load_translation_subset(subset_size=1000)
    tokenizer = load_tokenizer()
    batches   = build_batches(pairs, tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN)

    vocab_size = tokenizer.vocab_size
    print(f"\nVocabulário: {vocab_size:,} tokens")
    print(f"Batches    : {len(batches)} × batch_size={BATCH_SIZE}")

    # Instanciar modelo
    model = TransformerPyTorch(
        vocab_size = vocab_size,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF,
        max_len    = MAX_LEN + 2,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {n_params:,}")

    # Training Loop
    loss_history = train(model, batches)

    # Salvar modelo treinado para a Tarefa 4
    torch.save(model.state_dict(), "transformer_trained.pt")
    print(f"\n✓ Pesos salvos em 'transformer_trained.pt'")
    print("=" * 60)

    return model, tokenizer, pairs, loss_history


if __name__ == "__main__":
    demo()
