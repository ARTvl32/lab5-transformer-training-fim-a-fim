"""
Laboratório 5 — Tarefa 1: Preparando o Dataset Real (Hugging Face)
===================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Carrega o dataset Helsinki-NLP/opus_books (par en-fr) do Hugging Face e
seleciona as primeiras 1.000 frases como conjunto de treinamento.

O subconjunto minúsculo garante que o treinamento rode rapidamente na
CPU ou no Google Colab gratuito, sem exigir GPUs dedicadas.

Dependências
------------
    pip install datasets
"""

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATASET_NAME   = "Helsinki-NLP/opus_books"
DATASET_CONFIG = "en-fr"
SUBSET_SIZE    = 1000


# ---------------------------------------------------------------------------
# Tarefa 1 — Carregamento e subconjunto
# ---------------------------------------------------------------------------

def load_translation_subset(
    dataset_name=DATASET_NAME,
    config=DATASET_CONFIG,
    subset_size=SUBSET_SIZE,
):
    """
    Carrega o dataset de tradução do Hugging Face e retorna as primeiras
    `subset_size` frases como lista de dicionários {"en": ..., "fr": ...}.

    Parâmetros
    ----------
    dataset_name : str   — nome do dataset no Hugging Face Hub
    config       : str   — par de línguas (ex: "en-fr")
    subset_size  : int   — número máximo de pares a carregar

    Retorna
    -------
    pairs : list[dict]
        Lista de dicionários com chaves "en" e "fr".
        Ex: [{"en": "A man walks.", "fr": "Un homme marche."}, ...]
    """
    print(f"Carregando dataset '{dataset_name}' ({config})...")
    dataset = load_dataset(dataset_name, config, split=f"train[:{subset_size}]")

    # O campo "translation" contém {"en": ..., "fr": ...}
    pairs = [sample["translation"] for sample in dataset]

    print(f"✓ {len(pairs)} pares de frases carregados.")
    return pairs


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 1 — Dataset Real (Hugging Face)")
    print("=" * 60)

    pairs = load_translation_subset()

    print(f"\nPrimeiros 5 pares de frases:")
    for i, pair in enumerate(pairs[:5]):
        print(f"\n  [{i+1}]")
        print(f"    EN: {pair['en']}")
        print(f"    FR: {pair['fr']}")

    # Estatísticas básicas
    en_lens = [len(p["en"].split()) for p in pairs]
    fr_lens = [len(p["fr"].split()) for p in pairs]
    print(f"\nEstatísticas do subset ({len(pairs)} pares):")
    print(f"  Comprimento médio EN : {sum(en_lens)/len(en_lens):.1f} palavras")
    print(f"  Comprimento médio FR : {sum(fr_lens)/len(fr_lens):.1f} palavras")
    print(f"  Comprimento máx. EN  : {max(en_lens)} palavras")
    print(f"  Comprimento máx. FR  : {max(fr_lens)} palavras")

    print("\n✓ Dataset pronto para a Tarefa 2 (tokenização).")
    print("=" * 60)

    return pairs


if __name__ == "__main__":
    demo()
