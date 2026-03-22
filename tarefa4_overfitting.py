"""
Laboratório 5 — Tarefa 4: A Prova de Fogo (Overfitting Test)
=============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Técnica clássica de debugging de redes neurais: forçar o modelo a decorar
um conjunto ínfimo de dados para provar que os gradientes fluem corretamente.

Se a arquitetura estiver correta, o modelo deve conseguir memorizar um único
par de frases após poucas épocas de treinamento intenso, reproduzindo a
tradução exata via loop auto-regressivo (implementado na Tarefa 4 do Lab 04).

Etapas
------
1. Selecionar 1 frase específica do conjunto de treino
2. Treinar o modelo exclusivamente nessa frase por 200 épocas
3. Executar o loop auto-regressivo com essa frase como entrada
4. Verificar que a tradução gerada corresponde (ou é muito próxima) à
   tradução real — provando que os gradientes fluíram corretamente

Por que isso funciona?
----------------------
Um modelo com capacidade suficiente DEVE ser capaz de memorizar 1 exemplo.
Se não consegue, indica problema no fluxo de gradientes (bug arquitetural).
Se consegue, prova que toda a cadeia Forward→Loss→Backward→Step está correta.

Dependências
------------
    pip install torch transformers datasets
"""

import torch
import torch.nn as nn
import math
from tarefa2_tokenizacao import load_tokenizer, PAD_ID, START_ID, EOS_ID
from tarefa3_training_loop import (
    TransformerPyTorch,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN, DEVICE,
)


# ---------------------------------------------------------------------------
# Frase de teste (hardcoded para garantir reprodutibilidade)
# ---------------------------------------------------------------------------

TEST_PAIR = {
    "en": "A man is walking in the park.",
    "fr": "Un homme marche dans le parc.",
}

OVERFIT_EPOCHS = 200
OVERFIT_LR     = 5e-4   # lr maior para convergência mais rápida em 1 exemplo


# ---------------------------------------------------------------------------
# Overfitting em 1 par
# ---------------------------------------------------------------------------

def overfit_single_pair(model, tokenizer, pair, epochs=OVERFIT_EPOCHS, lr=OVERFIT_LR):
    """
    Treina o modelo exclusivamente sobre 1 par de frases por `epochs` épocas.

    Parâmetros
    ----------
    model     : TransformerPyTorch  — modelo já instanciado
    tokenizer : AutoTokenizer
    pair      : dict {"en": str, "fr": str}
    epochs    : int
    lr        : float

    Retorna
    -------
    loss_history : list[float]
    """
    model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tokenizar o único par
    en_ids = tokenizer.encode(pair["en"], add_special_tokens=False, max_length=MAX_LEN, truncation=True)
    fr_ids = tokenizer.encode(pair["fr"], add_special_tokens=False, max_length=MAX_LEN, truncation=True)

    src     = torch.tensor([en_ids],              dtype=torch.long).to(DEVICE)  # (1, T_src)
    dec_inp = torch.tensor([[START_ID] + fr_ids], dtype=torch.long).to(DEVICE)  # (1, T_tgt)
    tgt     = torch.tensor([fr_ids + [EOS_ID]],   dtype=torch.long).to(DEVICE)  # (1, T_tgt)

    loss_history = []
    print_steps  = {1, 10, 25, 50, 100, 150, 200}

    print(f"\nOverfitting em 1 par por {epochs} épocas...")
    print(f"  EN: '{pair['en']}'")
    print(f"  FR: '{pair['fr']}'")
    print()

    for epoch in range(1, epochs + 1):
        logits = model(src, dec_inp)                         # (1, T_tgt, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tgt.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch in print_steps:
            print(f"  Época {epoch:3d}/{epochs} | Loss: {loss_val:.6f}")

    print(f"\n✓ Loss inicial : {loss_history[0]:.4f}")
    print(f"✓ Loss final   : {loss_history[-1]:.6f}")

    return loss_history


# ---------------------------------------------------------------------------
# Loop auto-regressivo (portado do Lab 04 para PyTorch)
# ---------------------------------------------------------------------------

def autoregressive_inference(model, tokenizer, src_text, max_new_tokens=50):
    """
    Executa o loop auto-regressivo de inferência para gerar a tradução.

    Inicia com <START>, chama decode() iterativamente, aplica argmax,
    concatena o novo token ao contexto do Decoder e para ao gerar <EOS>.

    Parâmetros
    ----------
    model      : TransformerPyTorch
    tokenizer  : AutoTokenizer
    src_text   : str  — frase de entrada em EN
    max_new_tokens : int

    Retorna
    -------
    translated : str  — tradução gerada pelo modelo
    """
    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        # Tokenizar entrada
        src_ids = tokenizer.encode(
            src_text, add_special_tokens=False,
            max_length=MAX_LEN, truncation=True,
        )
        src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

        # Encoder processa a entrada UMA vez
        Z = model.encode(src)

        # Decoder começa com <START>
        decoder_ids = [START_ID]

        generated_ids = []
        for step in range(max_new_tokens):
            dec_input = torch.tensor([decoder_ids], dtype=torch.long).to(DEVICE)
            logits    = model.decode(dec_input, Z)        # (1, T_dec, V)

            # Pegar apenas a predição do último token
            next_id = logits[0, -1, :].argmax().item()

            if next_id == EOS_ID:
                break

            decoder_ids.append(next_id)
            generated_ids.append(next_id)

    translated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return translated


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("TAREFA 4 — Prova de Fogo (Overfitting Test)")
    print("=" * 65)

    tokenizer  = load_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Instanciar modelo novo (sem pesos pré-treinados)
    model = TransformerPyTorch(
        vocab_size = vocab_size,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF,
        max_len    = MAX_LEN + 2,
    )

    # --- Fase 1: Inferência ANTES do treinamento ---
    print("\n--- Inferência ANTES do overfitting ---")
    pred_before = autoregressive_inference(model, tokenizer, TEST_PAIR["en"])
    print(f"  Entrada   : {TEST_PAIR['en']}")
    print(f"  Esperado  : {TEST_PAIR['fr']}")
    print(f"  Gerado    : '{pred_before}'  (aleatório — pesos iniciais)")

    # --- Fase 2: Overfitting em 1 par ---
    loss_history = overfit_single_pair(model, tokenizer, TEST_PAIR)

    # --- Fase 3: Inferência DEPOIS do treinamento ---
    print("\n--- Inferência DEPOIS do overfitting ---")
    pred_after = autoregressive_inference(model, tokenizer, TEST_PAIR["en"])
    print(f"  Entrada   : {TEST_PAIR['en']}")
    print(f"  Esperado  : {TEST_PAIR['fr']}")
    print(f"  Gerado    : '{pred_after}'")

    # --- Avaliação ---
    print("\n--- Avaliação ---")
    expected_ids  = tokenizer.encode(TEST_PAIR["fr"], add_special_tokens=False)
    generated_ids = tokenizer.encode(pred_after,      add_special_tokens=False)

    # Overlap token a token
    matches = sum(a == b for a, b in zip(expected_ids, generated_ids))
    total   = max(len(expected_ids), 1)
    overlap = matches / total * 100

    print(f"  Overlap token-a-token: {matches}/{total} ({overlap:.1f}%)")
    print(f"  Loss caiu para: {loss_history[-1]:.6f}")

    if loss_history[-1] < 0.1:
        print("\n✓ PROVA DE FOGO APROVADA: Loss < 0.1, gradientes fluindo corretamente.")
    elif loss_history[-1] < 1.0:
        print("\n~ PROVA DE FOGO PARCIAL: Loss < 1.0, convergência em andamento.")
    else:
        print("\n✗ ATENÇÃO: Loss ainda alto. Verificar fluxo de gradientes.")

    print("=" * 65)


if __name__ == "__main__":
    demo()
