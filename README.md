# NLP Microservice (FastAPI)

Минимальный NLP-микросервис на FastAPI с реализациями Bag-of-Words, TF‑IDF и LSA (через TruncatedSVD), а также базовыми операциями NLTK для русского языка.

## Структура проекта

```
project/
  README.md
  requirements.txt
  .gitignore
  data/
    README.md
    raw/
    corpus.txt
  client/
    __init__.py
    download_corpus.py
    run_client.py
    client_utils.py
  server/
    __init__.py
    main.py
    routes.py
    schemas.py
    models.py
    nltk_ops.py
    utils.py
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Загрузка корпуса

```bash
python client/download_corpus.py
```

Скачивается UD Russian GSD (CC BY-SA 4.0), `.conllu` сохраняются в `data/raw/`, а маленький корпус (300 предложений) — в `data/corpus.txt`.

Источник корпуса: [UD Russian GSD](https://universaldependencies.org/treebanks/ru_gsd/index.html), лицензия CC BY-SA 4.0.

## Запуск сервера

```bash
uvicorn server.main:app --reload
```

## Запуск клиента

```bash
python client/run_client.py
```

## Примеры запросов

```bash
curl -X POST http://127.0.0.1:8000/bag-of-words \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Пример текста"], "lower": true, "min_token_len": 2}'
```

```bash
curl -X POST http://127.0.0.1:8000/tf-idf \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Пример текста"], "smooth_idf": true, "normalize": "l2"}'
```

```bash
curl -X POST http://127.0.0.1:8000/text_nltk/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Привет, это тест"}'
```

## Коротко о формулах

**TF‑IDF**:
- `TF = counts / max(1, sum(counts по документу))`
- `DF = (counts > 0).sum(axis=0)`
- `IDF = log((1 + N) / (1 + DF)) + 1` (сглаженная версия)
- `TFIDF = TF * IDF`, далее L2-нормировка строк через `np.linalg.norm`.

**LSA**:
- TF‑IDF матрица по документам
- `TruncatedSVD(n_components=...)` для получения тематических компонент и эмбеддингов документов.
