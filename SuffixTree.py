#!/usr/bin/env python3
import os
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Utilitare
# =========================

def compute_similarity(text1: str, text2: str) -> float:
    """Returnează cosine-similarity TF-IDF între două secțiuni."""
    vecs = TfidfVectorizer().fit_transform([text1, text2])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])

# =========================
# Ukkonen's Suffix Tree
# =========================
class End:
    def __init__(self, value): self.value = value

class Node:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.children = {}
        self.suffix_link = None

    def edge_length(self, pos):
        return min(self.end.value, pos + 1) - self.start

class SuffixTree:
    def __init__(self, text: str):
        self.text = text
        self.root = Node(-1, End(-1))
        self.root.suffix_link = self.root
        self._build()

    def _build(self):
        text = self.text
        size = len(text)
        root = self.root
        active_node = root
        active_edge = -1
        active_length = 0
        remaining = 0
        end = End(-1)
        last_new = None
        for i in range(size):
            end.value += 1
            remaining += 1
            last_new = None
            while remaining > 0:
                if active_length == 0:
                    active_edge = i
                ch = text[active_edge]
                if ch not in active_node.children:
                    active_node.children[ch] = Node(i, end)
                    if last_new:
                        last_new.suffix_link = active_node
                        last_new = None
                else:
                    nxt = active_node.children[ch]
                    edge_len = nxt.edge_length(i)
                    if active_length >= edge_len:
                        active_edge += edge_len
                        active_length -= edge_len
                        active_node = nxt
                        continue
                    if text[nxt.start + active_length] == text[i]:
                        active_length += 1
                        if last_new:
                            last_new.suffix_link = active_node
                            last_new = None
                        break
                    # split
                    split = Node(nxt.start, End(nxt.start + active_length - 1))
                    active_node.children[ch] = split
                    split.children[text[i]] = Node(i, end)
                    nxt.start += active_length
                    split.children[text[nxt.start]] = nxt
                    if last_new:
                        last_new.suffix_link = split
                    last_new = split
                remaining -= 1
                if active_node == root and active_length > 0:
                    active_length -= 1
                    active_edge = i - remaining + 1
                else:
                    active_node = active_node.suffix_link or root

# =========================
# Matching și clusterizare
# =========================

def find_matches(tree: SuffixTree, susp: str, min_len: int) -> list[tuple[int,int,int]]:
    matches = []
    text = tree.text
    n_s = len(susp)
    for i in range(n_s):
        node = tree.root
        j = i
        length = 0
        src_off = -1
        while j < n_s:
            ch = susp[j]
            if ch not in node.children:
                break
            edge = node.children[ch]
            span = edge.edge_length(j)
            k = 0
            while k < span and j < n_s and text[edge.start + k] == susp[j]:
                if length == 0:
                    src_off = edge.start + k
                length += 1
                j += 1
                k += 1
            if k < span:
                break
            node = edge
        if length >= min_len:
            # ajustăm src_off la început de segment
            start_src = src_off - (length - 1)
            matches.append((i, length, start_src))
    return matches


def cluster_and_filter(matches: list[tuple[int,int,int]], min_len: int) -> list[tuple[int,int,int]]:
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
    return [c for c in clusters if c[1] >= min_len]

# =========================
# Scriere XML
# =========================

def write_xml(matches, susp_name, src_name, susp_path, src_path, out_path, authors, title, lang, sim_score, severity):
    root = ET.Element('document', reference=susp_name)
    # about
    ET.SubElement(root, 'feature', {
        'name':       'about',
        'authors':    authors,
        'title':      title,
        'lang':       lang,
        'similarity': f"{sim_score:.4f}",
        'severity':   severity,
        'algorithm':  'SuffixTree+Filter'
    })
    # md5
    text_susp = open(susp_path, 'r', encoding='utf-8').read()
    ET.SubElement(root, 'feature', {
        'name':  'md5Hash',
        'value': hashlib.md5(text_susp.encode('utf-8')).hexdigest()
    })
    # segmente
    src_text = open(src_path, 'r', encoding='utf-8').read()
    for this_off, length, src_off in matches:
        seg_s = text_susp[this_off:this_off+length]
        seg_x = src_text[src_off:src_off+length]
        local_sim = compute_similarity(seg_s, seg_x)
        ET.SubElement(root, 'feature', {
            'name':             'detected-plagiarism',
            'type':             'suffix-tree',
            'this_language':    lang,
            'this_offset':      str(this_off),
            'this_length':      str(length),
            'source_reference': src_name,
            'source_offset':    str(src_off),
            'source_length':    str(length),
            'local_similarity': f"{local_sim:.3f}"
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)

# =========================
# Metadata
# =========================

def parse_metadata(file_path: str) -> tuple[str, str]:
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

# =========================
# Main
# =========================

def main():
    base_dir   = '00_spot_check'
    out_dir    = 'results-SuffixTreeBetter'
    os.makedirs(out_dir, exist_ok=True)
    pairs_file = os.path.join(base_dir, 'pairs')

    # parametri
    min_len       = 30   # prag redus pentru recall
    lang          = 'en'
    local_thresh  = 0.6  # pragmatic filter

    trees = {}

    for line in open(pairs_file, 'r', encoding='utf-8'):
        line=line.strip()
        if not line or line.startswith('#'): continue
        susp_name, src_name = line.split()
        susp_path = os.path.join(base_dir, 'susp', susp_name)
        src_path  = os.path.join(base_dir, 'src',  src_name)
        out_path  = os.path.join(out_dir, f"{susp_name[:-4]}-{src_name[:-4]}.xml")

        print(f"Processing {susp_name} vs {src_name} (min_len={min_len}, local_thresh={local_thresh})...")
        title, authors = parse_metadata(susp_path)
        susp_text = open(susp_path,'r',encoding='utf-8').read()
        if src_name not in trees:
            src_text = open(src_path,'r',encoding='utf-8').read()
            trees[src_name] = SuffixTree(src_text)
        tree = trees[src_name]

        raw = find_matches(tree, susp_text, min_len)
        print(f"  raw matches: {len(raw)}")
        clustered = cluster_and_filter(raw, min_len)
        print(f"  clustered: {len(clustered)}")

        # filtru local
        filtered = []
        src_text = open(src_path,'r',encoding='utf-8').read()
        for m in clustered:
            s_off, length, x_off = m
            seg_s = susp_text[s_off:s_off+length]
            seg_x = src_text[x_off:x_off+length]
            sim = compute_similarity(seg_s, seg_x)
            if sim >= local_thresh:
                filtered.append(m)
        print(f"  after local filter: {len(filtered)}")

        sim_score = compute_similarity(susp_text, open(src_path,'r',encoding='utf-8').read())
        severity = 'high' if len(filtered)>100 else 'medium' if filtered else 'low'

        write_xml(filtered, susp_name, src_name, susp_path, src_path, out_path,
                  authors, title, lang, sim_score, severity)
        print(f" → Wrote {out_path} (global sim={sim_score:.4f})\n")

if __name__=='__main__':
    main()