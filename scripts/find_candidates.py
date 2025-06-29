#!/usr/bin/env python
"""
find_candidates.py · v4  (2025-06-28)

Adds a fuzzy fallback so acronym-style subreddit names (‘als’, ‘cmt’)
still hit even if ORDO lacks that abbreviation.

Usage
-----
python scripts/find_candidates.py          # scan only
python scripts/find_candidates.py --verify # scan + GPT filter
"""

import argparse, pathlib, json, csv, io, hashlib, subprocess, sys
import tqdm, spacy, zstandard as zstd
from rapidfuzz import process, fuzz

# ───────── configuration ─────────────────────────────────────────── #
SPACY_MODEL   = "models/model_rare_disease"
ORDO_TSV      = pathlib.Path("data/ordo_terms.tsv")
DUMP_DIR      = pathlib.Path("data/raw_subreddits")
OUT_CSV       = pathlib.Path("data/candidate_subreddits.csv")
CHUNK         = 1_000_000          # flush every million lines
FUZZY_THRESH  = 85                 # 0-100 RapidFuzz score
# ─────────────────────────────────────────────────────────────────── #

# load ORDO term list for fuzzy fallback
ORDO_TERMS = []
with ORDO_TSV.open(encoding="utf-8") as f:
    next(f)                                # skip header
    for line in f:
        _, term, _ = line.rstrip("\n").split("\t")
        ORDO_TERMS.append(term)

def pick_files():
    meta = sorted(DUMP_DIR.rglob("*meta_only*.zst"))
    return meta if meta else sorted(DUMP_DIR.rglob("*.zst"))

def open_zst(path: pathlib.Path):
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    raw  = dctx.stream_reader(open(path, "rb"))
    return io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")

def hash_line(line: str) -> str:
    return hashlib.md5(line.encode("utf-8")).hexdigest()

def flush(rows):
    if not rows:
        return 0
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return len(rows)

def scan() -> bool:
    nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "lemmatizer"])
    hit_ids = lambda txt: {e.ent_id_ for e in nlp(txt).ents}

    files = pick_files()
    if not files:
        raise SystemExit(f"No .zst files found in {DUMP_DIR}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["subreddit", "orpha_ids"])

    seen_hash, total_kept = set(), 0
    for fn in files:
        print(f"→ scanning {fn.name}")
        rows, processed = [], 0

        with open_zst(fn) as fh:
            for line in tqdm.tqdm(fh, unit=" lines", mininterval=1.0):
                processed += 1
                obj = json.loads(line)

                raw_name = obj.get("name", "")
                name     = raw_name[2:] if raw_name.startswith("r/") else raw_name
                txt      = f"{name} {obj.get('title','')} {obj.get('public_description','')}".lower()

                ids = hit_ids(txt)

                # fuzzy backup if EntityRuler missed and name is ≥4 chars
                if not ids and len(name) >= 3:
                    term, score = process.extractOne(
                        name.lower(), ORDO_TERMS, scorer=fuzz.partial_ratio
                    )
                    if score >= FUZZY_THRESH:
                        ids = {"FUZZY"}          # marker; ORPHA unknown

                if ids:
                    h = hash_line(raw_name)
                    if h not in seen_hash:
                        rows.append([raw_name, ";".join(sorted(ids))])
                        seen_hash.add(h)

                if processed % CHUNK == 0:
                    total_kept += flush(rows)
                    rows.clear()
                    print(f" flushed at {processed:,} lines")

        total_kept += flush(rows)

    print(f"[✓] {total_kept} candidate subreddits → {OUT_CSV}")
    return total_kept > 0

def run_verify():
    print("\n🚀  Launching OpenAI verification…\n")
    subprocess.run([sys.executable, "scripts/verify_subreddits_openai.py"], check=True)

# ─────────────────────────── entry point ─────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="run OpenAI filter after scan completes")
    args = parser.parse_args()

    if scan() and args.verify:
        run_verify()





