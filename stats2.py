import os
import xml.etree.ElementTree as ET
import pandas as pd

# Mapare folder → coloană Excel
folders = {
    "Adevăr":                        "00_spot_check_truth",
    "SuffixAutomaton":               "results-SuffixAutomaton",
    "SuffixAutomaton Improved":      "results-SuffixAutomaton-improved",
}

# Alege unul dintre foldere pentru listarea fișierelor (ar trebui să conțină aceleași 50 de nume .xml)
example_folder = folders["SuffixAutomaton"]
xml_files = sorted(f for f in os.listdir(example_folder) if f.endswith(".xml"))

# Inițializează DataFrame cu index Document 1..N
n = len(xml_files)
df = pd.DataFrame(index=range(1, n+1))
df.index.name = "Document"

# Extrage pentru fiecare coloană și fișier valoarea similarity
for col_name, folder in folders.items():
    sims = []
    for fname in xml_files:
        path = os.path.join(folder, fname)
        try:
            tree  = ET.parse(path)
            about = tree.getroot().find('feature[@name="about"]')
            sim   = about.get("similarity") if about is not None else None
        except Exception:
            sim = None
        sims.append(float(sim) if sim is not None else None)
    df[col_name] = sims

# Adaugă rândul cu medii pe fiecare coloană
df.loc["Average"] = df.mean(numeric_only=True)

# Scrie fișierul Excel
output_file = "comparison_suffixautomaton.xlsx"
df.to_excel(output_file)

print(f"Saved comparison table to {output_file}")
