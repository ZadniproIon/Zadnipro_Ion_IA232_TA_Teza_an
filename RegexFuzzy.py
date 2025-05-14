#!/usr/bin/env python3
import os
import hashlib
import xml.etree.ElementTree as ET
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ––––– utilitare –––––

def parse_metadata(file_path: str) -> tuple[str, str]:
    """Extrage titlul și autorii din fișierul text:
    primul rând non-vid este titlul, următorul rând care începe cu 'By ' conține autorii."""
    title = ''
    authors = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not title:
                title = line
                continue
            if line.lower().startswith('by '):
                authors = line[3:].strip()
                break
    return title, authors


def md5_hash(text: str) -> str:
    """Returnează MD5 hex digest pentru un text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def compute_similarity(text1: str, text2: str) -> float:
    """Cosine similarity TF-IDF între două documente."""
    vecs = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def find_fuzzy_matches(susp: str, src: str, min_len: int, max_err: int) -> list[tuple[int,int,int]]:
    """Folosește regex fuzzy pentru a găsi în src porțiuni asemănătoare cu fragmente din susp.
    returnează liste de (susp_offset, length, src_offset)."""
    matches = []
    step = max(1, min_len // 2)
    for i in range(0, len(susp) - min_len + 1, step):
        snippet = susp[i:i+min_len]
        # construim pattern cu până la max_err erori peste snippet
        pattern = regex.compile(rf'({regex.escape(snippet)}){{e<={max_err}}}', flags=regex.BESTMATCH)
        m = pattern.search(src)
        if m:
            matches.append((i, min_len, m.start()))
    return matches


def write_xml(
    matches: list[tuple[int,int,int]],
    susp_name: str,
    src_name: str,
    susp_path: str,
    out_path: str,
    authors: str,
    title: str,
    lang: str,
    sim_score: float,
    severity: str,
    algorithm: str
):
    root = ET.Element('document', reference=susp_name)
    # metadate generale
    ET.SubElement(root, 'feature', {
        'name':       'about',
        'authors':    authors,
        'title':      title,
        'lang':       lang,
        'similarity': f"{sim_score:.4f}",
        'severity':   severity,
        'algorithm':  algorithm
    })
    # hash MD5
    with open(susp_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()
    ET.SubElement(root, 'feature', {
        'name':  'md5Hash',
        'value': md5_hash(doc_text)
    })
    # segmente detectate
    for this_off, length, src_off in matches:
        ET.SubElement(root, 'feature', {
            'name':             'plagiarism',
            'type':             'fuzzy-match',
            'this_language':    lang,
            'this_offset':      str(this_off),
            'this_length':      str(length),
            'source_reference': src_name,
            'source_offset':    str(src_off),
            'source_length':    str(length)
        })
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main():
    base_dir   = '00_spot_check'
    pairs_file = os.path.join(base_dir, 'pairs')
    result_dir = 'results-RegexFuzzy'
    os.makedirs(result_dir, exist_ok=True)

    # parametri prototip
    min_len = 50        # lungime minimă de potrivire
    max_err = 2         # număr maxim de erori permise
    lang    = 'en'
    algorithm = 'RegexFuzzy'

    with open(pairs_file, 'r', encoding='utf-8') as pf:
        for line in pf:
            pair = line.strip()
            if not pair or pair.startswith('#'):
                continue
            susp_name, src_name = pair.split()
            susp_path = os.path.join(base_dir, 'susp', susp_name)
            src_path  = os.path.join(base_dir, 'src', src_name)
            out_path  = os.path.join(result_dir, f"{susp_name[:-4]}-{src_name[:-4]}.xml")

            # afișăm progresul
            print(f"Processing: {susp_name} vs {src_name}...")

            # extragem metadatele din fișier
            title, authors = parse_metadata(susp_path)
            # citim textele
            src_text  = open(src_path, 'r', encoding='utf-8').read()
            susp_text = open(susp_path, 'r', encoding='utf-8').read()

            # detectăm segmente fuzzy și scor global
            matches   = find_fuzzy_matches(susp_text, src_text, min_len, max_err)
            sim_score = compute_similarity(susp_text, src_text)
            # determinăm severitatea
            if sim_score >= 0.99:
                severity = 'high'
            elif sim_score >= 0.95:
                severity = 'medium'
            else:
                severity = 'low'

            # salvăm XML-ul
            write_xml(
                matches, susp_name, src_name,
                susp_path, out_path,
                authors, title,
                lang, sim_score,
                severity, algorithm
            )
            print(f" -> Done: {len(matches)} segments, sim={sim_score:.4f}\n")

if __name__ == '__main__':
    main()
