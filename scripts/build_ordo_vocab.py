#!/usr/bin/env python
"""
build_ordo_vocab.py  ·  v1.3  (2025-06-27)

• Downloads the latest Orphanet Rare Disease Ontology (ORDO)
• Extracts English labels + synonyms  →  data/ordo_terms.tsv
• Saves a spaCy EntityRuler          →  models/model_rare_disease/

Run once, then rerun only when ORDO releases a new major version.
"""

import pathlib, rdflib, unicodedata, re, csv, requests, tqdm, spacy, sys, textwrap

# -------------------------------------------------------------------- #
#  1.  Where can we get ORDO?   (checked 27 Jun 2025)
# -------------------------------------------------------------------- #
ORDO_CANDIDATES = [
    # a) Orphadata “official” ZIP (directory listing verified) :contentReference[oaicite:0]{index=0}
    ("https://www.orphadata.com/ordo_orphanet_4.5.owl.zip", "zip"),
    # b) Orphadata SPARQL page (raw OWL) – sometimes allowed
    ("https://www.orphadata.com/ordo/ORDO_en_4.5.owl", "owl"),
    # c) OBO PURL mirror
    ("https://purl.obolibrary.org/obo/ordo.owl", "owl"),
    # d) GitHub fallback (community mirror)
    ("https://raw.githubusercontent.com/laiasubirats/rarediseasesontology/master/ordo_orphanet.owl",
     "owl"),
]

DATA_DIR  = pathlib.Path("data")
MODEL_DIR = pathlib.Path("models") / "model_rare_disease"


# -------------------------------------------------------------------- #
#  2.  Helpers
# -------------------------------------------------------------------- #
def normalise(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text)
    txt = re.sub(r"[‐–—−]", "-", txt)          # unify dashes
    return txt.lower()


def download_owl(dest: pathlib.Path):
    if dest.exists():
        print(f"[✓] ORDO already on disk → {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "rare-reddit-pipeline/1.0"})
    for url, kind in ORDO_CANDIDATES:
        try:
            print(f"[+] Trying {url}")
            r = sess.get(url, timeout=120)
            r.raise_for_status()
            if kind == "zip":
                import zipfile, io
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    # grab the first *.owl file inside
                    owl_name = next(x for x in zf.namelist() if x.endswith(".owl"))
                    dest.write_bytes(zf.read(owl_name))
            else:
                dest.write_bytes(r.content)
            print("[✓] ORDO downloaded successfully")
            return
        except Exception as e:
            print(f"[!] {type(e).__name__}: {e} – trying next mirror")

    # fall-through → nothing worked
    print(
        textwrap.dedent(f"""
        ───────────────────────────────────────────────────────────────
        ❌  All automated downloads failed.

        1. Visit the Orphadata page in a browser and grab the file:
           https://www.orphadata.com/ordo/   (look for “ordo_orphanet_4.5.owl.zip”)

        2. Place the extracted .owl here:
           {dest}

        3. Re-run:  python scripts/build_ordo_vocab.py
        ───────────────────────────────────────────────────────────────
        """).strip()
    )
    sys.exit(1)


def extract_terms(owl_path: pathlib.Path):
    g = rdflib.Graph().parse(str(owl_path))
    LABEL = rdflib.RDFS.label
    ALT   = rdflib.URIRef("http://www.ebi.ac.uk/efo/alternative_term")

    rows = []
    for s, _, label in tqdm.tqdm(g.triples((None, LABEL, None)), desc="labels"):
        if not label.language or label.language.startswith("en"):
            rows.append((s.split("_")[-1], normalise(str(label)), 1))
    for s, _, syn in tqdm.tqdm(g.triples((None, ALT, None)), desc="synonyms"):
        rows.append((s.split("_")[-1], normalise(str(syn)), 0))
    return rows


def write_tsv(rows, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["orpha_id", "term", "is_preferred"])
        w.writerows(rows)
    print(f"[✓] Wrote {len(rows):,} rows → {dest}")


def build_spacy_model(rows):
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"validate": True})
    ruler.add_patterns([
        {"label": "RARE_DISEASE", "id": f"ORPHA{oid}", "pattern": term}
        for oid, term, _ in rows
    ])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"[✓] Saved spaCy model → {MODEL_DIR}")


# -------------------------------------------------------------------- #
#  3.  Main
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    owl_file = DATA_DIR / "ORDO_latest.owl"
    download_owl(owl_file)
    terms = extract_terms(owl_file)
    write_tsv(terms, DATA_DIR / "ordo_terms.tsv")
    build_spacy_model(terms)
