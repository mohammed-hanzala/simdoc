"""SimDoc similarity engine.

Four parallel similarity modules -> learned-style sigmoid ensemble (Eq. 1 in the paper):

    y_hat = sigmoid(w0 + sum_i w_i * s_i)

Two modes:
  * lite (default): no model downloads. TF-IDF is the real paper baseline; the three
    "neural" slots use lightweight approximations (subword, order-aware, concept-level).
  * full (SIMDOC_MODE=full): real bert-base-uncased, roberta-base and SBERT via
    sentence-transformers (pip install -r requirements-full.txt).
"""
import math
import os
import re
import time
from collections import OrderedDict
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODE = os.environ.get("SIMDOC_MODE", "lite").lower()
MAX_SENTENCES = 14

# ---------------------------------------------------------------- preprocessing
STOP = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "it its this that these those i you he she we they them his her their our your my not no "
    "so than then too very can will would should could do does did has have had into over "
    "about after before up down out off per via".split()
)

# Tiny thesaurus: every word in a group maps to one concept id (lite "semantic" module).
SYN_GROUPS = [
    "car automobile vehicle auto", "big large huge enormous massive great major wide broad", "small little tiny minor",
    "quick fast rapid speedy swift", "buy purchase acquire", "begin start commence launch initiate",
    "end finish conclude complete terminate", "show display reveal present demonstrate announce reveal",
    "help assist aid support", "use utilize employ apply", "make create build produce generate construct",
    "good great excellent fine superb", "bad poor terrible awful", "smart intelligent clever bright",
    "city urban town metropolitan metro area", "government authority authorities administration ministry state",
    "reduce cut lower decrease lessen shrink", "increase raise grow rise expand boost",
    "pollution smog contamination emission", "factory plant industrial industry mill",
    "citizen people person resident individual commuter", "encourage urge advise recommend persuade",
    "transport transit transportation commute travel driving drive", "bus train metro rail subway",
    "new fresh novel recent latest", "step measure action policy initiative rule",
    "install fit mount deploy equip", "filter filtration purifier scrubber",
    "year twelvemonth annual", "must require need obliged compulsory mandatory",
    "document text file paper article passage", "similar alike comparable analogous related",
    "model network algorithm system method approach technique", "accurate precise exact correct",
    "student pupil learner", "teacher instructor professor lecturer educator",
    "doctor physician clinician", "illness disease sickness ailment condition",
    "happy glad joyful cheerful pleased", "sad unhappy sorrowful gloomy",
    "say tell state declare mention remark", "look watch see view observe",
    "rain rainfall shower downpour", "heavy intense severe strong torrential",
    "mumbai bombay", "money cash fund currency", "company firm business corporation enterprise",
    "kid child children youngster", "movie film picture cinema", "music song tune melody",
    "photo image picture photograph", "road street avenue highway", "house home dwelling residence",
    "copy duplicate replicate clone plagiarize", "change alter modify adjust revise edit",
    "problem issue challenge difficulty trouble", "answer reply respond response solution",
    "cook bake prepare roast fry", "recipe dish meal food", "game match contest tournament",
    "win victory triumph defeat beat", "score point run goal",
    "बारिश वर्षा", "तेज़ भारी तीव्र ज़ोरदार", "आज आजकल",
]


def stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suf in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


CONCEPT = {}
for gi, grp in enumerate(SYN_GROUPS):
    for w in grp.split():
        CONCEPT[w] = CONCEPT[stem(w)] = f"c{gi}"


def tokens(text: str):
    return re.findall(r"[\wऀ-ॿ]+", text.lower())


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    return [clean(p) for p in parts if len(clean(p)) > 2]


def concept_string(text: str) -> str:
    out = []
    for t in tokens(text):
        if t in STOP:
            continue
        out.append(CONCEPT.get(t) or CONCEPT.get(stem(t)) or stem(t))
    return " ".join(out)


# ---------------------------------------------------------------- modules
def _cos_tfidf(vec, corpus, a_texts, b_texts):
    try:
        vec.fit(corpus)
        A, B = vec.transform(a_texts), vec.transform(b_texts)
        return cosine_similarity(A, B)
    except ValueError:  # empty vocabulary
        return np.zeros((len(a_texts), len(b_texts)))


class TfidfModule:
    key, label, real = "tfidf", "TF-IDF + cosine", True
    desc = "Sparse bag-of-words, unigrams + bigrams, sublinear TF (paper §IV-A)"

    def matrix(self, A, B):
        corpus = A + B
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              max_df=0.95 if len(corpus) >= 20 else 1.0,
                              token_pattern=r"[\wऀ-ॿ]+")
        return _cos_tfidf(vec, corpus, A, B)


class SubwordModule:
    key, label, real = "bert", "BERT (pooled)", False
    desc = "Lite stand-in: character 3-5-gram TF-IDF, a subword-style view"

    def matrix(self, A, B):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        return _cos_tfidf(vec, A + B, A, B)


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


class OrderModule:
    key, label, real = "roberta", "RoBERTa (pooled)", False
    desc = "Lite stand-in: word-order aware (bigram Dice + longest common subsequence)"

    @staticmethod
    def _pair(a, b):
        ta = [stem(t) for t in tokens(a) if t not in STOP][:300]
        tb = [stem(t) for t in tokens(b) if t not in STOP][:300]
        if not ta or not tb:
            return 0.0
        sa, sb = set(ta), set(tb)
        jac = len(sa & sb) / len(sa | sb)
        ba, bb = set(zip(ta, ta[1:])), set(zip(tb, tb[1:]))
        dice = (2 * len(ba & bb) / (len(ba) + len(bb))) if (ba or bb) else (1.0 if ta == tb else 0.0)
        lcs = _lcs(ta, tb) / max(len(ta), len(tb))
        return 0.25 * jac + 0.35 * dice + 0.40 * lcs

    def matrix(self, A, B):
        return np.array([[self._pair(a, b) for b in B] for a in A])


class ConceptModule:
    key, label, real = "sbert", "SBERT", False
    desc = "Lite stand-in: stemmed, synonym-normalised concept TF-IDF"

    def matrix(self, A, B):
        ca, cb = [concept_string(t) for t in A], [concept_string(t) for t in B]
        vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True)
        return _cos_tfidf(vec, ca + cb, ca, cb)


class EmbedModule:
    """Real transformer module (full mode)."""
    real = True

    def __init__(self, key, label, hf_name, desc):
        self.key, self.label, self.hf_name, self.desc = key, label, hf_name, desc
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer, models
            if self.key == "sbert":
                self._model = SentenceTransformer(self.hf_name)
            else:  # raw BERT / RoBERTa with mean pooling, as in the paper
                tr = models.Transformer(self.hf_name, max_seq_length=256)
                pool = models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean")
                self._model = SentenceTransformer(modules=[tr, pool])
        return self._model

    def matrix(self, A, B):
        m = self._load()
        ea = m.encode(A, normalize_embeddings=True)
        eb = m.encode(B, normalize_embeddings=True)
        return np.clip(ea @ eb.T, 0, 1)


def build_modules():
    if MODE == "full":
        return [
            TfidfModule(),
            EmbedModule("bert", "BERT (pooled)", "bert-base-uncased", "bert-base-uncased, mean-pooled (paper §IV-B)"),
            EmbedModule("roberta", "RoBERTa (pooled)", "roberta-base", "roberta-base, mean-pooled (paper §IV-B)"),
            EmbedModule("sbert", "SBERT", "sentence-transformers/all-MiniLM-L6-v2", "Pre-trained SBERT checkpoint (paper §IV-C)"),
        ]
    return [TfidfModule(), SubwordModule(), OrderModule(), ConceptModule()]


MODULES = build_modules()

# ---------------------------------------------------------------- ensemble (Eq. 1)
# Hand-set demo weights (NOT the paper's trained values). The paper learns these on the
# STS-B train split; lite-mode proxies score lower than real transformers, so lite mode
# uses a gentler bias and leans harder on the concept module.
if MODE == "full":
    BIAS = -4.0
    WEIGHTS = {"tfidf": 1.2, "bert": 1.9, "roberta": 2.2, "sbert": 2.9}
else:
    BIAS = -1.6
    WEIGHTS = {"tfidf": 1.5, "bert": 2.0, "roberta": 1.5, "sbert": 4.5}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def verdict(score: float):
    if score >= 0.80:
        return "Near-duplicate", "dup"
    if score >= 0.60:
        return "Highly similar", "high"
    if score >= 0.40:
        return "Related", "mid"
    return "Different", "low"


# ---------------------------------------------------------------- cache (Redis stand-in)
class TTLCache:
    def __init__(self, ttl=24 * 3600, size=500):
        self.ttl, self.size, self.d = ttl, size, OrderedDict()
        self.hits = self.misses = 0

    def get(self, k):
        v = self.d.get(k)
        if v and time.time() - v[0] < self.ttl:
            self.d.move_to_end(k)
            self.hits += 1
            return v[1]
        self.misses += 1
        return None

    def put(self, k, val):
        self.d[k] = (time.time(), val)
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return round(self.hits / t, 3) if t else 0.0


CACHE = TTLCache()


def compare(doc_a: str, doc_b: str):
    key = hashlib.sha1(f"{MODE}\x00{doc_a}\x00{doc_b}".encode()).hexdigest()
    hit = CACHE.get(key)
    if hit:
        return {**hit, "cached": True}

    t0 = time.perf_counter()
    sa_all, sb_all = split_sentences(doc_a), split_sentences(doc_b)
    sa, sb = sa_all[:MAX_SENTENCES], sb_all[:MAX_SENTENCES]
    A = [clean(doc_a)] + sa
    B = [clean(doc_b)] + sb

    mats, models = {}, []
    for m in MODULES:
        t1 = time.perf_counter()
        M = np.clip(np.nan_to_num(m.matrix(A, B)), 0, 1)
        ms = (time.perf_counter() - t1) * 1000
        mats[m.key] = M
        s = float(M[0, 0])
        w = WEIGHTS[m.key]
        models.append({"key": m.key, "label": m.label, "desc": m.desc, "approx": not m.real,
                       "score": round(s, 4), "weight": w, "contribution": round(w * s, 4),
                       "latency_ms": round(ms, 1)})

    logit = BIAS + sum(x["contribution"] for x in models)
    final = float(sigmoid(logit))

    # sentence-level alignment with the same ensemble
    align = BIAS + sum(WEIGHTS[k] * mats[k] for k in mats)
    S = sigmoid(align)[1:, 1:] if (sa and sb) else np.zeros((len(sa), len(sb)))
    best = []
    for i in range(len(sa)):
        if len(sb):
            j = int(np.argmax(S[i]))
            best.append({"a": i, "b": j, "score": round(float(S[i, j]), 4)})

    label, tier = verdict(final)
    res = {
        "score": round(final, 4), "verdict": label, "tier": tier, "mode": MODE,
        "bias": BIAS, "logit": round(float(logit), 4), "models": models,
        "alignment": {"a": sa, "b": sb, "matrix": np.round(S, 3).tolist(), "best": best,
                      "truncated": len(sa_all) > MAX_SENTENCES or len(sb_all) > MAX_SENTENCES},
        "stats": {"words_a": len(tokens(doc_a)), "words_b": len(tokens(doc_b)),
                  "shared_terms": len(set(tokens(doc_a)) & set(tokens(doc_b)))},
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "cached": False,
    }
    CACHE.put(key, res)
    return res
