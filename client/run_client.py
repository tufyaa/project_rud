"""Run demo requests against the FastAPI service."""
from __future__ import annotations

from pathlib import Path
from client.client_utils import post_json, pretty_print

BASE_URL = "http://127.0.0.1:8000"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.txt"


def main() -> None:
    """Send sample requests to the API."""
    texts = DATA_PATH.read_text(encoding="utf-8").splitlines()[:50]

    bow_payload = {"texts": texts, "lower": True, "min_token_len": 2}
    bow_data = post_json(f"{BASE_URL}/bag-of-words", bow_payload)
    pretty_print("Bag of Words", bow_data)

    tfidf_payload = {
        "texts": texts,
        "smooth_idf": True,
        "normalize": "l2",
        "lower": True,
        "min_token_len": 2,
    }
    tfidf_data = post_json(f"{BASE_URL}/tf-idf", tfidf_payload)
    pretty_print("TF-IDF", tfidf_data)

    lsa_payload = {
        "texts": texts,
        "n_components": 2,
        "n_top_terms": 10,
        "lower": True,
        "min_token_len": 2,
    }
    lsa_data = post_json(f"{BASE_URL}/lsa", lsa_payload)
    pretty_print("LSA", lsa_data)

    text_payload = {"text": "Привет, это тестовый текст для NLTK и Natasha."}
    for endpoint, title in [
        ("tokenize", "Tokenize"),
        ("stem", "Stem"),
        ("lemmatize", "Lemmatize"),
        ("pos", "POS"),
        ("ner", "NER"),
    ]:
        data = post_json(f"{BASE_URL}/text_nltk/{endpoint}", text_payload)
        pretty_print(title, data)


if __name__ == "__main__":
    main()
