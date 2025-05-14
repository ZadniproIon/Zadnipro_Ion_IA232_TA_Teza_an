#!/usr/bin/env python3
import os
import hashlib
import xml.etree.ElementTree as ET
from collections import deque, defaultdict

# =========================
# Winnowing (Fingerprinting)
# =========================

def rolling_hash(s: str, base: int = 256, mod: int = 2**64) -> int:
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h

def winnow(text: str, k: int = 7, w: int = 5) -> list[tuple[int,int]]:
    n = len(text)
    if n < k:
        return []
    hashes = [rolling_hash(text[i:i+k]) for i in range(n - k + 1)]
    dq = deque()
    fingerprints = []
    for i, h in enumerate(hashes):
        while dq and dq[-1][0] >= h:
            dq.pop()
        dq.append((h, i))
        if i >= w and dq[0][1] <= i - w:
            dq.popleft()
        if i >= w - 1:
            fingerprints.append((dq[0][0], dq[0][1]))
    # elimin duplicate consecutive
    res, prev = [], None
    for f in fingerprints:
        if f != prev:
            res.append(f)
        prev = f
    return res

def find_winnow_matches(susp: str, src: str, k: int, w: int) -> list[tuple[int,int,int]]:
    fp_s = winnow(susp, k, w)
    fp_x = winnow(src,   k, w)
    src_map = defaultdict(list)
    for h, pos in fp_x:
        src_map[h].append(pos)

    matches, seen = [], set()
    for h, spos in fp_s:
        for xpos in src_map.get(h, []):
            key = (spos, xpos)
            if key not in seen:
                seen.add(key)
                matches.append((spos, k, xpos))
    return matches

def merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]:
    """Unifică intervalele suprapuse și returnează o listă de (start,end)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def cluster_and_filter(matches: list[tuple[int,int,int]], min_len: int = 50) -> list[tuple[int,int,int]]:
    if not matches:
        return []
    matches = sorted(set(matches), key=lambda x: x[0])
    clusters = []
    cs, length, xs = matches[0]
    ce = cs + length
    for spos, length, xpos in matches[1:]:
        e = spos + length
        if spos <= ce + 1 and (xpos - spos) == (xs - cs):
            ce = max(ce, e)
        else:
            clusters.append((cs, ce - cs, xs))
            cs, ce, xs = spos, e, xpos
    clusters.append((cs, ce - cs, xs))
    return [seg for seg in clusters if seg[1] >= min_len]

def write_xml(
    matches, susp_name, src_name,
    susp_path, out_path,
    authors, title, lang,
    severity, similarity
):
    root = ET.Element('document', reference=susp_name)
    ET.SubElement(root, 'feature', {
        'name':       'about',
        'authors':    authors,
        'title':      title,
        'lang':       lang,
        'severity':   severity,
        'similarity': f"{similarity:.4f}",
        'algorithm':  'Winnowing'
    })
    with open(susp_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ET.SubElement(root, 'feature', {
        'name':  'md5Hash',
        'value': hashlib.md5(text.encode('utf-8')).hexdigest()
    })
    for this_off, length, src_off in matches:
        ET.SubElement(root, 'feature', {
            'name':             'detected-plagiarism',
            'type':             'winnowing',
            'this_language':    lang,
            'this_offset':      str(this_off),
            'this_length':      str(length),
            'source_reference': src_name,
            'source_offset':    str(src_off),
            'source_length':    str(length)
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)

def parse_metadata(file_path):
    title = ''
    authors = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if not l: continue
            if not title:
                title = l
            elif l.lower().startswith('by '):
                authors = l[3:].strip()
                break
    return title, authors

def main():
    base_dir   = '00_spot_check'
    pairs_file = os.path.join(base_dir, 'pairs')
    out_dir    = 'results-Winnowing'
    os.makedirs(out_dir, exist_ok=True)

    k, w, min_len = 7, 5, 50
    lang = 'en'

    for line in open(pairs_file, 'r', encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        susp_name, src_name = line.split()
        susp_path = os.path.join(base_dir, 'susp', susp_name)
        src_path  = os.path.join(base_dir, 'src',  src_name)
        out_path  = os.path.join(out_dir, f"{susp_name[:-4]}-{src_name[:-4]}.xml")

        print(f"Processing {susp_name} vs {src_name}...")

        title, authors = parse_metadata(susp_path)
        with open(susp_path, 'r', encoding='utf-8') as f: susp_text = f.read()
        with open(src_path,  'r', encoding='utf-8') as f: src_text  = f.read()

        raw_matches = find_winnow_matches(susp_text, src_text, k, w)
        print(f"  raw fingerprints: {len(raw_matches)}")

        matches = cluster_and_filter(raw_matches, min_len)
        print(f"  clusters ≥{min_len} chars: {len(matches)}")

        severity = 'high' if len(matches) > 100 else 'medium' if matches else 'low'

        # === noul calcul de similarity ===
        # construim intervale [spos, spos+k) și le unim
        intervals = [(spos, spos + k) for spos, k, _ in raw_matches]
        merged = merge_intervals(intervals)
        coverage = sum(end - start for start, end in merged)
        similarity = coverage / max(len(susp_text), 1)

        write_xml(
            matches, susp_name, src_name,
            susp_path, out_path,
            authors, title, lang,
            severity, similarity
        )
        print(f" → Wrote {out_path} (similarity={similarity:.4f})\n")

if __name__ == '__main__':
    main()
