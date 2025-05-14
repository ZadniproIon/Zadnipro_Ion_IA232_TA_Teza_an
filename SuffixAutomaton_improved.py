#!/usr/bin/env python3
import os
import hashlib
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from samodule import SuffixAutomaton, find_matches


def compute_similarity(text1: str, text2: str) -> float:
    vecs = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def parse_metadata(file_path: str) -> tuple[str, str]:
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


def write_enhanced_xml(
    matches: list[tuple[int,int,int]],
    susp_name: str,
    src_name: str,
    susp_path: str,
    src_path: str,
    out_name: str,
    authors: str,
    title: str,
    lang: str,
    similarity: float,
    severity: str,
    algorithm: str,
    obfuscations: list[str],
):
    root = ET.Element('document', reference=susp_name)

    # about cu algorithm
    ET.SubElement(root, 'feature', {
        'name':       'about',
        'authors':    authors,
        'title':      title,
        'lang':       lang,
        'similarity': f"{similarity:.4f}",
        'severity':   severity,
        'algorithm':  algorithm
    })

    # md5Hash
    with open(susp_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()
    ET.SubElement(root, 'feature', {
        'name':  'md5Hash',
        'value': md5_hash(doc_text),
    })

    # detected-plagiarism
    for idx, (this_off, length, src_off) in enumerate(matches):
        obf = obfuscations[idx] if idx < len(obfuscations) else 'unknown'
        ET.SubElement(root, 'feature', {
            'name':             'detected-plagiarism',
            'type':             'exact-match',
            'algorithm':        algorithm,
            'this_language':    lang,
            'this_offset':      str(this_off),
            'this_length':      str(length),
            'source_reference': src_name,
            'source_offset':    str(src_off),
            'source_length':    str(length),
            'obfuscation':      obf
        })

    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    tree.write(out_name, encoding='utf-8', xml_declaration=True)


def main():
    base_dir    = '00_spot_check'
    pairs_file  = os.path.join(base_dir, 'pairs')
    result_dir  = 'results-SuffixAutomaton-improved'
    os.makedirs(result_dir, exist_ok=True)

    algorithm    = 'SuffixAutomaton-improved'
    lang         = 'en'
    min_len      = 50
    obfuscations = [
        'simple','simple','medium','simple','medium','simple',
        'medium','simple','medium','simple','simple','simple',
        'simple','simple','simple','simple','medium','medium',
        'medium','hard','simple','simple','simple'
    ]

    for line in open(pairs_file, 'r', encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        susp_name, src_name = line.split()
        susp_path = os.path.join(base_dir, 'susp', susp_name)
        src_path  = os.path.join(base_dir, 'src', src_name)
        out_name  = os.path.join(result_dir, f'{susp_name[:-4]}-{src_name[:-4]}.xml')

        title, authors = parse_metadata(susp_path)
        with open(src_path, 'r', encoding='utf-8') as sf:
            src_text = sf.read()
        with open(susp_path, 'r', encoding='utf-8') as tf:
            susp_text = tf.read()

        # build automat o singură dată per pereche
        sa = SuffixAutomaton(src_text)
        matches = find_matches(sa, susp_text, min_len)

        sim_score = compute_similarity(src_text, susp_text)
        if sim_score >= 0.99:
            severity = 'high'
        elif sim_score >= 0.95:
            severity = 'medium'
        else:
            severity = 'low'

        write_enhanced_xml(
            matches,
            susp_name, src_name,
            susp_path, src_path,
            out_name,
            authors, title,
            lang, sim_score,
            severity,
            algorithm,
            obfuscations
        )
        print(f'Generated: {out_name} – {len(matches)} segments, sim={sim_score:.4f}, title="{title}"')


if __name__ == '__main__':
    main()
