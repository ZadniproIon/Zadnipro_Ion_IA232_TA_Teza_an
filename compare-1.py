#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

def load_segments(xml_path: str) -> set[tuple[int, int, int]]:
    """
    Parsează XML-ul dat și returnează setul de segmente de plagiat:
    fiecare element e un tuplu (this_offset, length, source_offset).
    Se filtrează doar <feature name="plagiarism">.
    """
    segments = set()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR parsing {xml_path}: {e}")
        return segments

    for feat in root.findall('feature'):
        if feat.get('name') == 'plagiarism':
            try:
                this_off = int(feat.get('this_offset', 0))
                length   = int(feat.get('this_length', 0))
                src_off  = int(feat.get('source_offset', 0))
                segments.add((this_off, length, src_off))
            except ValueError:
                # skip entries with invalid numbers
                continue
    return segments


def evaluate(truth: set, pred: set) -> tuple[int, int, int, float, float, float]:
    """
    Calculează TP, FP, FN, precision, recall și F1 între seturile de segmente.
    """
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def main():
    truth_dir = '00_spot_check_truth'
    suffix_dir = 'results-SuffixAutomaton'
    seq_dir    = 'results-SequenceMatcher'

    # găsește toate fișierele XML de ground-truth
    truth_files = [f for f in os.listdir(truth_dir) if f.endswith('.xml')]
    if not truth_files:
        print(f"No XML files found in {truth_dir}")
        return

    # acumulatoare pentru statistici globale
    stats = {
        'SuffixAutomaton': {'tp': 0, 'fp': 0, 'fn': 0},
        'SequenceMatcher': {'tp': 0, 'fp': 0, 'fn': 0},
    }

    print("Evaluating results against ground truth...")
    print("File                                 |  SA P     R     F1   |  SM P     R     F1")
    print('-'*85)

    for fname in sorted(truth_files):
        truth_path = os.path.join(truth_dir, fname)
        base = os.path.splitext(fname)[0]

        sa_path = os.path.join(suffix_dir, fname)
        sm_path = os.path.join(seq_dir,    fname)

        if not os.path.exists(sa_path) or not os.path.exists(sm_path):
            print(f"Missing results for {fname}")
            continue

        truth_segs = load_segments(truth_path)
        sa_segs    = load_segments(sa_path)
        sm_segs    = load_segments(sm_path)

        tp1, fp1, fn1, p1, r1, f11 = evaluate(truth_segs, sa_segs)
        tp2, fp2, fn2, p2, r2, f12 = evaluate(truth_segs, sm_segs)

        # actualizează globale
        stats['SuffixAutomaton']['tp'] += tp1
        stats['SuffixAutomaton']['fp'] += fp1
        stats['SuffixAutomaton']['fn'] += fn1
        stats['SequenceMatcher']['tp'] += tp2
        stats['SequenceMatcher']['fp'] += fp2
        stats['SequenceMatcher']['fn'] += fn2

        # formatare output per fișier
        print(f"{base:35s} | {p1:6.3f} {r1:6.3f} {f11:6.3f} | {p2:6.3f} {r2:6.3f} {f12:6.3f}")

    # statistici globale
    print('-'*85)
    for alg in ['SuffixAutomaton', 'SequenceMatcher']:
        tp = stats[alg]['tp']
        fp = stats[alg]['fp']
        fn = stats[alg]['fn']
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        print(f"Overall {alg:15s}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

if __name__ == '__main__':
    main()
