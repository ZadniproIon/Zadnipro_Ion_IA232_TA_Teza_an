import os
import xml.etree.ElementTree as ET
import pandas as pd

# Mapare folder → nume coloană Excel
folders = {
    "Adevăr":               "00_spot_check_truth",
    "Rabin–Karp":           "results-RabinKarp",
    "Regex Fuzzy":          "results-RegexFuzzy",
    "SequenceMatcher":      "results-SequenceMatcher",
    "Suffix Automaton":     "results-SuffixAutomaton",
    "Suffix Tree":          "results-SuffixTree",
    "Suffix Tree Better":   "results-SuffixTreeBetter",
    "Winnowing":            "results-Winnowing",
}

# Listează fișierele XML (ar trebui să fie aceleași 50 în fiecare folder)
example_folder = folders["Rabin–Karp"]
xml_files = sorted(f for f in os.listdir(example_folder) if f.endswith(".xml"))

# Creează DataFrame cu index Document = 1..N
doc_count = len(xml_files)
df = pd.DataFrame(index=range(1, doc_count+1))
df.index.name = "Document"

# Parcurge fiecare coloană/folder și citește similarity
for col_name, folder in folders.items():
    sims = []
    for fname in xml_files:
        path = os.path.join(folder, fname)
        try:
            tree = ET.parse(path)
            about = tree.getroot().find('feature[@name="about"]')
            sim = about.get("similarity") if about is not None else None
        except Exception:
            sim = None
        sims.append(float(sim) if sim is not None else None)
    df[col_name] = sims

# Adaugă un rând cu mediile pe fiecare coloană
df.loc['Average'] = df.mean(numeric_only=True)

# Scrie rezultatul în Excel
output_file = "similarity_results_with_average.xlsx"
df.to_excel(output_file)
print(f"Am salvat rezultatele în {output_file}")
