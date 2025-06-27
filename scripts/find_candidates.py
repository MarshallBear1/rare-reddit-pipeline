#!/usr/bin/env python
"""
find_candidates.py

Streams every *.jsonl*.zst* in data/raw_subreddits/,
labels subreddit titles + descriptions with the spaCy
EntityRuler you built, and writes
    data/candidate_subreddits.csv   (subreddit, orpha_ids)
"""

import pathlib, json, gzip, csv, io
import tqdm, spacy, zstandard as zstd

SPACY_MODEL = "models/model_rare_disease"
DUMP_DIR    = pathlib.Path("data/raw_subreddits")
OUT_CSV     = pathlib.Path("data/candidate_subreddits.csv")

nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "lemmatizer"])
def hit_ids(text: str):          # returns a set of ORPHA IDs
    return {ent.ent_id_ for ent in nlp(text).ents}

rows = []
for fn in sorted(DUMP_DIR.rglob("*.zst")):      # recurse just in case
    print(f"→ {fn.name}")
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    reader = dctx.stream_reader(open(fn, "rb"))
    with io.TextIOWrapper(reader, encoding="utf-8", errors="ignore") as fh:
        for line in tqdm.tqdm(fh, unit=" lines", mininterval=1.0, leave=False):
            obj  = json.loads(line)
            text = f"{obj.get('title','')} {obj.get('public_description','')}"
            ids  = hit_ids(text)
            if ids:
                rows.append([obj["name"], ";".join(sorted(ids))])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["subreddit", "orpha_ids"], *rows])

print(f"[✓] {len(rows):,} candidate subreddits → {OUT_CSV}")
