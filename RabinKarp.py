#!/usr/bin/env python3
import os
import hashlib
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Utilitare
# =========================

def compute_similarity(text1: str, text2: str) -> float:
    """Returnează cosine-similarity TF-IDF între două documente."""
    vecs = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])

# =========================
# Binary Search + str.find Detector
# =========================

def find_max_match(src: str, susp: str, start: int, min_len: int) -> tuple[int,int,int] | None:
    """
    Pe poziția 'start' din susp, găsește cea mai lungă subsecvență comună de lungime >= min_len
    folosind bin-search și str.find. Returnează (start, length, src_pos) sau None.
    """
    lo, hi = min_len, len(susp) - start
    best_len = 0
    best_pos = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        snippet = susp[start:start+mid]
        pos = src.find(snippet)
        if pos != -1:
            best_len = mid
            best_pos = pos
            lo = mid + 1
        else:
            hi = mid - 1
    if best_len >= min_len:
        return (start, best_len, best_pos)
    return None

# =========================
# Clusterizare și filtrare
# =========================

def cluster_and_filter(matches: list[tuple[int,int,int]], min_len: int) -> list[tuple[int,int,int]]:
    """Grupează matches adiacente în segmente și filtrează lungimea minimă."""
    if not matches:
        return []
    matches = sorted(set(matches), key=lambda x: x[0])
    clusters = []
    cs, length, xs = matches[0]
    ce = cs + length
    for spos, length, xpos in matches[1:]:
        end = spos + length
        if spos <= ce + 1 and (xpos - spos) == (xs - cs):
            ce = max(ce, end)
        else:
            clusters.append((cs, ce - cs, xs))
            cs, ce, xs = spos, end, xpos
    clusters.append((cs, ce - cs, xs))
    return [c for c in clusters if c[1] >= min_len]

# =========================
# XML Writing
# =========================

def write_xml(matches, susp_name, src_name, susp_path, src_path, out_path, authors, title, lang, sim_score, severity):
    root = ET.Element('document', reference=susp_name)
    ET.SubElement(root, 'feature', {
        'name':       'about',
        'authors':    authors,
        'title':      title,
        'lang':       lang,
        'similarity': f"{sim_score:.4f}",
        'severity':   severity,
        'algorithm':  'BinSearchFind'
    })
    text_susp = open(susp_path, 'r', encoding='utf-8').read()
    ET.SubElement(root, 'feature', {
        'name':  'md5Hash',
        'value': hashlib.md5(text_susp.encode('utf-8')).hexdigest()
    })
    for this_off, length, src_off in matches:
        ET.SubElement(root, 'feature', {
            'name':             'detected-plagiarism',
            'type':             'rabin-find',
            'this_language':    lang,
            'this_offset':      str(this_off),
            'this_length':      str(length),
            'source_reference': src_name,
            'source_offset':    str(src_off),
            'source_length':    str(length)
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)

# =========================
# Metadata Parsing
# =========================

def parse_metadata(file_path: str) -> tuple[str, str]:
    title = ''
    authors = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            if not title:
                title = l
            elif l.lower().startswith('by '):
                authors = l[3:].strip()
                break
    return title, authors

# =========================
# Main Batch
# =========================

def main():
    base_dir   = '00_spot_check'
    pairs_file = os.path.join(base_dir, 'pairs')
    out_dir    = 'results-RabinKarp'
    os.makedirs(out_dir, exist_ok=True)

    min_len = 30
    lang    = 'en'

    for line in open(pairs_file, 'r', encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        susp_name, src_name = line.split()
        susp_path = os.path.join(base_dir, 'susp', susp_name)
        src_path  = os.path.join(base_dir, 'src', src_name)
        out_path  = os.path.join(out_dir, f"{susp_name[:-4]}-{src_name[:-4]}.xml")

        print(f"Processing {susp_name} vs {src_name} (min_len={min_len})...")
        title, authors = parse_metadata(susp_path)
        susp_text = open(susp_path, 'r', encoding='utf-8').read()
        src_text  = open(src_path, 'r', encoding='utf-8').read()

        raw_matches = []
        n = len(susp_text)
        for i in range(0, n - min_len + 1):
            m = find_max_match(src_text, susp_text, i, min_len)
            if m:
                raw_matches.append(m)
        print(f"  raw matches: {len(raw_matches)}")

        matches = cluster_and_filter(raw_matches, min_len)
        print(f"  clusters: {len(matches)}")

        sim_score = compute_similarity(susp_text, src_text)
        severity = 'high' if len(matches) > 100 else 'medium' if matches else 'low'

        write_xml(matches, susp_name, src_name, susp_path, src_path,
                  out_path, authors, title, lang, sim_score, severity)
        print(f" → Wrote {out_path} (sim={sim_score:.4f})\n")

if __name__ == '__main__':
    main()
