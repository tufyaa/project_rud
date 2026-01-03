from __future__ import annotations

from pathlib import Path
import requests

RAW_URLS = {
    "train": "https://github.com/UniversalDependencies/UD_Russian-GSD/raw/master/ru_gsd-ud-train.conllu",
    "dev": "https://github.com/UniversalDependencies/UD_Russian-GSD/raw/master/ru_gsd-ud-dev.conllu",
    "test": "https://github.com/UniversalDependencies/UD_Russian-GSD/raw/master/ru_gsd-ud-test.conllu",
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_PATH = DATA_DIR / "corpus.txt"


def download_file(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def parse_conllu_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current_tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current_tokens:
                sentences.append(" ".join(current_tokens))
                current_tokens = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        token_id = parts[0]
        if "-" in token_id or "." in token_id:
            continue
        form = parts[1]
        current_tokens.append(form)
    if current_tokens:
        sentences.append(" ".join(current_tokens))
    return sentences


def build_corpus(sentences: list[str], limit: int = 300) -> list[str]:
    return sentences[:limit]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_sentences: list[str] = []

    for split, url in RAW_URLS.items():
        dest = RAW_DIR / f"ru_gsd-ud-{split}.conllu"
        print(f"Downloading {split}...")
        download_file(url, dest)
        sentences = parse_conllu_sentences(dest.read_text(encoding="utf-8"))
        all_sentences.extend(sentences)

    corpus_sentences = build_corpus(all_sentences, limit=300)
    CORPUS_PATH.write_text("\n".join(corpus_sentences), encoding="utf-8")
    print(f"Saved corpus with {len(corpus_sentences)} sentences to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
