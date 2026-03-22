"""
Laboratório 5 — Tarefa 2: Tokenização Básica
=============================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Converte os pares de frases (texto) em listas de inteiros (IDs) usando o
tokenizador pré-treinado bert-base-multilingual-cased do Hugging Face.

O Transformer não lê texto — ele lê matrizes de números. Esta tarefa faz a
ponte entre o dataset bruto (Tarefa 1) e o modelo (Tarefa 3).

Etapas
------
1. Carregar AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
2. Iterar pelos 1.000 pares e converter cada frase em lista de IDs
3. Para as frases de destino (Decoder):
   - Adicionar [CLS] (101) no início como <START>
   - Adicionar [SEP] (102) no final como <EOS>
4. Aplicar padding com zeros para igualar o comprimento dentro do batch

Tokens especiais do bert-base-multilingual-cased
-------------------------------------------------
    PAD   = 0    (preenchimento)
    [CLS] = 101  → usado como <START>
    [SEP] = 102  → usado como <EOS>

Dependências
------------
    pip install transformers torch
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "bert-base-multilingual-cased"
MAX_LEN        = 64    # truncar frases acima deste comprimento
PAD_ID         = 0
START_ID       = 101   # [CLS] do BERT → <START>
EOS_ID         = 102   # [SEP] do BERT → <EOS>


# ---------------------------------------------------------------------------
# Tarefa 2 — Tokenização e padding
# ---------------------------------------------------------------------------

def load_tokenizer(name=TOKENIZER_NAME):
    """Carrega o tokenizador pré-treinado."""
    print(f"Carregando tokenizador '{name}'...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(f"✓ Tokenizador carregado. Vocab size: {tokenizer.vocab_size:,}")
    return tokenizer


def tokenize_pairs(pairs, tokenizer, max_len=MAX_LEN):
    """
    Converte uma lista de pares {"en": ..., "fr": ...} em listas de IDs.

    Para a frase de destino (FR / Decoder), adiciona:
        - START_ID no início  →  decoder_input:  [START, tok1, tok2, ...]
        - EOS_ID   no final   →  target_output:  [tok1, tok2, ..., EOS]

    Parâmetros
    ----------
    pairs     : list[dict]  — pares {"en": str, "fr": str}
    tokenizer : AutoTokenizer
    max_len   : int         — comprimento máximo (trunca se necessário)

    Retorna
    -------
    src_ids      : list[list[int]]  — IDs das frases EN (Encoder)
    dec_inp_ids  : list[list[int]]  — IDs [START + FR] (Decoder input)
    tgt_ids      : list[list[int]]  — IDs [FR + EOS]   (Target / labels)
    """
    src_ids, dec_inp_ids, tgt_ids = [], [], []

    for pair in pairs:
        # Tokenizar sem tokens especiais do BERT (usamos os nossos)
        en_ids = tokenizer.encode(
            pair["en"],
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
        )
        fr_ids = tokenizer.encode(
            pair["fr"],
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
        )

        # Encoder input: apenas os IDs da frase EN
        src_ids.append(en_ids)

        # Decoder input:  <START> + frase FR  (teacher forcing — entrada)
        dec_inp_ids.append([START_ID] + fr_ids)

        # Target output:  frase FR + <EOS>    (o que o modelo deve prever)
        tgt_ids.append(fr_ids + [EOS_ID])

    return src_ids, dec_inp_ids, tgt_ids


def pad_batch(sequences, pad_id=PAD_ID):
    """
    Aplica padding em uma lista de listas de IDs para que todas tenham o
    mesmo comprimento, preenchendo com pad_id.

    Parâmetros
    ----------
    sequences : list[list[int]]
    pad_id    : int

    Retorna
    -------
    padded : torch.LongTensor, shape (batch_size, max_seq_len)
    """
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    padded  = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
    return padded


def build_batches(pairs, tokenizer, batch_size=32, max_len=MAX_LEN):
    """
    Tokeniza todos os pares e organiza em batches com padding.

    Retorna
    -------
    batches : list[dict]
        Cada elemento: {"src": Tensor, "dec_inp": Tensor, "tgt": Tensor}
        Todos com shape (batch_size, seq_len).
    """
    src_ids, dec_inp_ids, tgt_ids = tokenize_pairs(pairs, tokenizer, max_len)

    batches = []
    for i in range(0, len(src_ids), batch_size):
        src_b     = pad_batch(src_ids[i:i+batch_size])
        dec_inp_b = pad_batch(dec_inp_ids[i:i+batch_size])
        tgt_b     = pad_batch(tgt_ids[i:i+batch_size])
        batches.append({"src": src_b, "dec_inp": dec_inp_b, "tgt": tgt_b})

    return batches


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("TAREFA 2 — Tokenização Básica")
    print("=" * 65)

    # Dataset de exemplo (em produção viria da Tarefa 1)
    sample_pairs = [
        {"en": "A man is walking in the park.",
         "fr": "Un homme marche dans le parc."},
        {"en": "The cat sat on the mat.",
         "fr": "Le chat était assis sur le tapis."},
        {"en": "Thinking machines learn from data.",
         "fr": "Les machines pensantes apprennent des données."},
    ]

    tokenizer = load_tokenizer()

    src_ids, dec_inp_ids, tgt_ids = tokenize_pairs(sample_pairs, tokenizer)

    print(f"\n--- Exemplo de tokenização (par 1) ---")
    print(f"  EN (texto)       : {sample_pairs[0]['en']}")
    print(f"  EN (IDs)         : {src_ids[0]}")
    print(f"\n  FR (texto)       : {sample_pairs[0]['fr']}")
    print(f"  dec_input (IDs)  : {dec_inp_ids[0]}  ← [START=101] + FR")
    print(f"  target    (IDs)  : {tgt_ids[0]}  ← FR + [EOS=102]")

    # Padding
    src_padded     = pad_batch(src_ids)
    dec_inp_padded = pad_batch(dec_inp_ids)
    tgt_padded     = pad_batch(tgt_ids)

    print(f"\n--- Após padding (batch de {len(sample_pairs)} frases) ---")
    print(f"  src_padded shape     : {src_padded.shape}")
    print(f"  dec_inp_padded shape : {dec_inp_padded.shape}")
    print(f"  tgt_padded shape     : {tgt_padded.shape}")

    print(f"\n  src_padded (tensor):\n{src_padded}")
    print(f"\n  Tokens PAD=0 preenchem até o comprimento máximo do batch ✓")

    # Verificações
    assert dec_inp_padded[0, 0].item() == START_ID, "dec_input deve começar com START"
    last_real = (tgt_padded[0] != PAD_ID).nonzero(as_tuple=True)[0][-1]
    assert tgt_padded[0, last_real].item() == EOS_ID, "target deve terminar com EOS"
    print(f"\n✓ dec_input começa com START_ID={START_ID}")
    print(f"✓ target termina com EOS_ID={EOS_ID}")
    print(f"✓ Padding com PAD_ID={PAD_ID} aplicado corretamente")
    print("=" * 65)

    return tokenizer, sample_pairs


if __name__ == "__main__":
    demo()
