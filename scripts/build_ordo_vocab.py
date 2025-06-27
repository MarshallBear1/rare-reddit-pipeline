#!/usr/bin/env python
"""
build_ordo_vocab.py

Download the latest ORDO (rare-disease ontology), extract English labels
+ synonyms, write them to data/ordo_terms.tsv, and save a spaCy
EntityRuler model in models/model_rare_disease/.

Run once, then re-run only when ORDO updates.
"""

import pathlib, rdflib, unicodedata, re, csv, requests, tqdm, spacy

# --- CONFIG ---------------------------------------------------------

ORDO_URLS = [
    # Mirror 1 – Orphadata “last_version” folder
    "https://www.orphadata.com/data/ontologies/ordo/last_version/ORDO_en_4.5.owl",
    # Mirror 2 – OBO PURL
    "https://raw.githubusercontent.com/obophenotype/ORDO/main/ordo.owl",
]

DATA_DIR  = pathlib.Path("data")
MODEL_DIR = pathlib.Path("models") / "model_rare_disease"

# --------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, Unicode-normalise, unify dashes."""
    txt = unicodedata.normalize("NFKD", text)
    txt = re.sub(r"[‐--–—−]", "-", txt)
    return txt.lower()

def download_owl(dest: pathlib.Path):
    if dest.exists():
        print(f"[✓] ORDO already on disk → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in ORDO_URLS:
        try:
            print(f"[+] Fetching {url}")
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print("[✓] Downloaded ORDO")
            return
        except requests.HTTPError as e:
            print(f"[!] {e} – trying next mirror")
    raise RuntimeError("All ORDO mirrors failed – aborting.")

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

# --------------------------------------------------------------------

if __name__ == "__main__":
    owl_file = DATA_DIR / "ORDO_latest.owl"
    download_owl(owl_file)
    term_rows = extract_terms(owl_file)
    write_tsv(term_rows, DATA_DIR / "ordo_terms.tsv")
    build_spacy_model(term_rows)
