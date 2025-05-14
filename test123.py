import os
import hashlib
import xml.etree.ElementTree as ET
from samodule import SuffixAutomaton, find_matches
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import spacy

# === Load spaCy model for lemmatization ===
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess(text: str) -> str:
    """Lemmatizează textul și elimină stop words."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.lemma_ not in ENGLISH_STOP_WORDS and token.is_alpha]
    return " ".join(tokens)

def adaptive_min_len(text: str) -> int:
    """Calculează dinamic pragul min_len în funcție de lungimea textului."""
    return max(50, int(len(text) / 1000))

def compute_similarity(text1: str, text2: str) -> float:
    """Cosine similarity TF-IDF între texte."""
    vecs = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])

def merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]:
    """Unifică intervalele suprapuse."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]

def find_fuzzy_matches(src: str, segment: str, threshold: float = 0.75) -> list[tuple[int,int,int]]:
    """
    Aplică SequenceMatcher pe un segment neacoperit.
    Returnează tupluri (start_in_susp, length, start_in_src) dacă similaritatea
    blocului ≥ threshold.
    """
    sm = SequenceMatcher(None, src, segment)
    matches = []
    for block in sm.get_matching_blocks():
        if block.size > 0:
            sim = block.size / len(segment)
            if sim >= threshold:
                matches.append((block.b, block.size, block.a))
    return matches

def write_xml(matches, susp_name, src_name, susp_path, src_path, out_path,
              authors, title, lang, sim_score, severity, algorithm):
    root = ET.Element("document", reference=susp_name)
    ET.SubElement(root, "feature", {
        "name":       "about",
        "authors":    authors,
        "title":      title,
        "lang":       lang,
        "similarity": f"{sim_score:.4f}",
        "severity":   severity,
        "algorithm":  algorithm
    })
    with open(susp_path, "r", encoding="utf-8") as f:
        txt = f.read()
    ET.SubElement(root, "feature", {
        "name":  "md5Hash",
        "value": hashlib.md5(txt.encode("utf-8")).hexdigest()
    })
    for this_off, length, src_off in matches:
        ET.SubElement(root, "feature", {
            "name":             "detected-plagiarism",
            "type":             "hybrid",
            "algorithm":        algorithm,
            "this_language":    lang,
            "this_offset":      str(this_off),
            "this_length":      str(length),
            "source_reference": src_name,
            "source_offset":    str(src_off),
            "source_length":    str(length)
        })
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)

def main():
    base = "00_spot_check"
    pairs = os.path.join(base, "pairs")
    out_dir = "results-SuffixAutomaton-improved-2"
    os.makedirs(out_dir, exist_ok=True)

    algorithm = "SuffixAutomaton-Improved-2"
    lang = "en"

    for line in open(pairs, "r", encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        susp_name, src_name = line.split()
        susp_path = os.path.join(base, "susp", susp_name)
        src_path  = os.path.join(base, "src",  src_name)
        out_path  = os.path.join(out_dir, f"{susp_name[:-4]}-{src_name[:-4]}.xml")

        # metadata
        title = ""
        authors = ""
        for l in open(susp_path, "r", encoding="utf-8"):
            t = l.strip()
            if not title and t:
                title = t
            elif t.lower().startswith("by "):
                authors = t[3:].strip()
                break

        # load & preprocess
        src_text  = preprocess(open(src_path, "r", encoding="utf-8").read())
        susp_text = preprocess(open(susp_path, "r", encoding="utf-8").read())

        # exact matching
        sa = SuffixAutomaton(src_text)
        min_len = adaptive_min_len(susp_text)
        exact = find_matches(sa, susp_text, min_len)

        # determine unmatched intervals
        intervals = [(off, off+length) for off, length, _ in exact]
        gaps = []
        last_end = 0
        for start, end in merge_intervals(intervals):
            if start > last_end:
                gaps.append((last_end, start))
            last_end = end
        if last_end < len(susp_text):
            gaps.append((last_end, len(susp_text)))

        # fuzzy matching on gaps
        hybrid = exact.copy()
        for start, end in gaps:
            segment = susp_text[start:end]
            if len(segment) < min_len:
                continue
            fuzz = find_fuzzy_matches(src_text, segment, threshold=0.75)
            # adjust offsets back to original
            for off_s, length, off_x in fuzz:
                hybrid.append((start+off_s, length, off_x))

        # merge all matches
        merged = [(s, e-s) for s, e in merge_intervals([(m[0], m[0]+m[1]) for m in hybrid])
                  for m in hybrid if m[0] == s and m[1] == e-s]

        # compute similarity
        sim_score = compute_similarity(src_text, susp_text)
        severity = "high" if sim_score >= 0.99 else "medium" if sim_score >= 0.95 else "low"

        write_xml(merged, susp_name, src_name, susp_path, src_path, out_path,
                  authors, title, lang, sim_score, severity, algorithm)
        print(f"Generated {out_path}: {len(merged)} segments, sim={sim_score:.4f}")

if __name__ == "__main__":
    main()

