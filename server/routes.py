from __future__ import annotations

from fastapi import APIRouter

from server.schemas import LsaRequest, TextRequest, TfidfRequest, VectorizeRequest
from server.models import build_bow, build_lsa, build_tfidf
from server.nltk_ops import (
    ensure_nltk_resources,
    extract_entities,
    lemmatize_tokens,
    pos_tag_tokens,
    stem_tokens,
    tokenize_text,
)
from server.utils import to_jsonable, validate_text, validate_texts

router = APIRouter()


@router.get("/")
def root() -> dict[str, object]:
    return {
        "message": "NLP microservice powered by FastAPI and NLTK",
        "endpoints": [
            "/bag-of-words",
            "/tf-idf",
            "/lsa",
            "/text_nltk/tokenize",
            "/text_nltk/stem",
            "/text_nltk/lemmatize",
            "/text_nltk/pos",
            "/text_nltk/ner",
        ],
        "example": {
            "method": "POST",
            "path": "/bag-of-words",
            "body": {"texts": ["Пример текста"], "lower": True, "min_token_len": 2},
        },
    }


@router.post("/bag-of-words")
def bag_of_words(payload: VectorizeRequest) -> dict[str, object]:
    texts = validate_texts(payload.texts)
    data = build_bow(texts, lower=payload.lower, min_token_len=payload.min_token_len)
    return {"vocabulary": data["vocabulary"], "counts": to_jsonable(data["counts"])}


@router.post("/tf-idf")
def tf_idf(payload: TfidfRequest) -> dict[str, object]:
    texts = validate_texts(payload.texts)
    data = build_tfidf(
        texts,
        lower=payload.lower,
        min_token_len=payload.min_token_len,
        smooth_idf=payload.smooth_idf,
        normalize=payload.normalize,
    )
    return {
        "vocabulary": data["vocabulary"],
        "tfidf": to_jsonable(data["tfidf"]),
        "idf": to_jsonable(data["idf"]),
    }


@router.post("/lsa")
def lsa(payload: LsaRequest) -> dict[str, object]:
    texts = validate_texts(payload.texts)
    data = build_lsa(
        texts,
        lower=payload.lower,
        min_token_len=payload.min_token_len,
        n_components=payload.n_components,
        n_top_terms=payload.n_top_terms,
    )
    return {
        "vocabulary": data["vocabulary"],
        "doc_embeddings": to_jsonable(data["doc_embeddings"]),
        "components": to_jsonable(data["components"]),
        "top_terms": data["top_terms"],
    }


@router.post("/text_nltk/tokenize")
def text_tokenize(payload: TextRequest) -> dict[str, object]:
    text = validate_text(payload.text)
    return tokenize_text(text)


@router.post("/text_nltk/stem")
def text_stem(payload: TextRequest) -> dict[str, object]:
    text = validate_text(payload.text)
    ensure_nltk_resources()
    tokens = tokenize_text(text)["tokens"]
    return {"tokens": tokens, "stems": stem_tokens(tokens)}


@router.post("/text_nltk/lemmatize")
def text_lemmatize(payload: TextRequest) -> dict[str, object]:
    text = validate_text(payload.text)
    ensure_nltk_resources()
    tokens = tokenize_text(text)["tokens"]
    return {"tokens": tokens, "lemmas": lemmatize_tokens(tokens)}


@router.post("/text_nltk/pos")
def text_pos(payload: TextRequest) -> dict[str, object]:
    text = validate_text(payload.text)
    ensure_nltk_resources()
    tokens = tokenize_text(text)["tokens"]
    tags = pos_tag_tokens(tokens)
    return {"tokens": tokens, "pos": tags}


@router.post("/text_nltk/ner")
def text_ner(payload: TextRequest) -> dict[str, object]:
    text = validate_text(payload.text)
    entities = extract_entities(text)
    return {"entities": entities}
