#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    # 1️⃣ Verificăm că directoarele există
    truth_dir  = '00_spot_check_truth'
    det_dir    = 'results-SuffixTree'
    eval_script = 'pan12-plagiarism-detection-evaluation.py'

    for d in (truth_dir, det_dir):
        if not os.path.isdir(d):
            print(f"ERROR: nu există directorul „{d}”", file=sys.stderr)
            sys.exit(1)
    if not os.path.isfile(eval_script):
        print(f"ERROR: nu am găsit scriptul „{eval_script}”", file=sys.stderr)
        sys.exit(1)

    # 2️⃣ Construim comanda
    cmd = [
        sys.executable, eval_script,
        '--plag-path', truth_dir,
        '--det-path',  det_dir,
        '--plag-tag',  'plagiarism',
        '--det-tag',   'detected-plagiarism'
    ]

    print("=== Evaluare SuffixAutomaton ===")
    # 3️⃣ Rulăm comanda şi redirecţionăm output-ul direct în consolă
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Scriptul de evaluare s-a încheiat cu eroare (cod {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)

if __name__ == '__main__':
    main()
