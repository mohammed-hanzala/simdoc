"""SimDoc Flask API.  POST /similarity  ->  {score, models[], alignment{}, ...}"""
import os
from flask import Flask, jsonify, request, send_from_directory

import engine

DIST = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=None)


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


MAX_CHARS = 20000


@app.get("/api/health")
def health():
    return jsonify(status="ok", mode=engine.MODE, cache_hit_rate=engine.CACHE.hit_rate,
                   modules=[{"key": m.key, "label": m.label, "approx": not m.real} for m in engine.MODULES])


@app.post("/similarity")
@app.post("/api/similarity")
def similarity():
    data = request.get_json(silent=True) or {}
    a = (data.get("doc_a") or data.get("text1") or "").strip()
    b = (data.get("doc_b") or data.get("text2") or "").strip()
    if not a or not b:
        return jsonify(error="Provide both doc_a and doc_b."), 400
    if len(a) > MAX_CHARS or len(b) > MAX_CHARS:
        return jsonify(error=f"Each document must be under {MAX_CHARS} characters."), 413
    return jsonify(engine.compare(a, b))


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path):
    if os.path.isdir(DIST):
        full = os.path.join(DIST, path)
        if path and os.path.isfile(full):
            return send_from_directory(DIST, path)
        return send_from_directory(DIST, "index.html")
    return jsonify(message="SimDoc API running. Frontend folder not found."), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
