# SimDoc — Text Similarity & Semantic Document Alignment

A light, working implementation of the SimDoc paper: four similarity modules
(TF-IDF, BERT, RoBERTa, SBERT) fused by a sigmoid ensemble (Eq. 1), a Flask REST API
(`POST /similarity`) and a dashboard with score gauge, per-model breakdown,
sentence-alignment heatmap, and CSV export.

## Run (no npm, no build step)

```bash
pip install -r api/requirements.txt
python api/app.py          # open http://localhost:5000
```

## Modes

| Mode | How | What runs |
|------|-----|-----------|
| **lite** (default) | nothing to download | Real TF-IDF + three lightweight stand-ins (char n-grams, word-order matching, synonym-normalised concepts) |
| **full** | `pip install -r api/requirements-full.txt` then `SIMDOC_MODE=full python api/app.py` | Real `bert-base-uncased`, `roberta-base` and SBERT (`all-MiniLM-L6-v2`) via sentence-transformers (downloads models on first use) |

Honest notes
- Ensemble weights are **hand-set**, not the paper's trained values (the paper learns them on STS-B train). Edit `WEIGHTS`/`BIAS` in `api/engine.py`.
- Full mode is written but was not run in the build environment (no model downloads there). Raw BERT/RoBERTa cosines are high, so weights likely need retraining in that mode.
- Redis is replaced by an in-process 24 h TTL cache.

## API

```bash
curl -X POST localhost:5000/similarity -H "Content-Type: application/json" \
  -d '{"doc_a": "Heavy rain hit Mumbai today.", "doc_b": "Mumbai received intense rainfall."}'
```
Returns `score`, `verdict`, per-model `models[]` (score, weight, latency) and `alignment` (sentence matrix + best matches). `GET /api/health` shows mode and modules.

## Layout
```
api/       Flask app + similarity engine
frontend/  static dashboard (index.html, app.js, data.js, styles.css)
```
