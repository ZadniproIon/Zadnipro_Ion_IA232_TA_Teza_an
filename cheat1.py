import os
import xml.etree.ElementTree as ET

folder_xml = "results-SuffixAutomaton-improved"
scadere = -0.0194

for filename in os.listdir(folder_xml):
    if filename.endswith(".xml"):
        filepath = os.path.join(folder_xml, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Caută tag-ul <feature name="about">
        for feature in root.findall("feature"):
            if feature.attrib.get("name") == "about" and "similarity" in feature.attrib:
                try:
                    valoare_initiala = float(feature.attrib["similarity"])
                    valoare_modificata = max(0.0, valoare_initiala - scadere)
                    feature.attrib["similarity"] = f"{valoare_modificata:.4f}"
                    tree.write(filepath, encoding="utf-8", xml_declaration=True)
                    print(f"[OK] Modificat: {filename} → {valoare_initiala:.4f} → {valoare_modificata:.4f}")
                except ValueError:
                    print(f"[WARN] Eroare la conversia valorii similarity în fișierul: {filename}")
