class SuffixAutomaton:
    class State:
        __slots__ = ('link','len','next','firstpos')
        def __init__(self):
            self.link = -1
            self.len = 0
            self.next = {}        # tranziții: char → stare
            self.firstpos = -1    # cel mai mic „endpos” în sursă

    def __init__(self, s: str):
        self.states = [self.State()]  # starea 0 = rădăcină
        self.last = 0
        for i, c in enumerate(s):
            self._extend(c, i)

    def _extend(self, c: str, pos: int):
        p = self.last
        cur = len(self.states)
        st = self.State()
        st.len = self.states[p].len + 1
        st.firstpos = pos
        self.states.append(st)
        self.last = cur

        # 1) legăm tranzițiile cu caracterul c
        while p >= 0 and c not in self.states[p].next:
            self.states[p].next[c] = cur
            p = self.states[p].link

        if p == -1:
            st.link = 0
        else:
            q = self.states[p].next[c]
            if self.states[p].len + 1 == self.states[q].len:
                st.link = q
            else:
                # clonăm starea q
                clone = len(self.states)
                qc = self.State()
                qc.len = self.states[p].len + 1
                qc.next = self.states[q].next.copy()
                qc.link = self.states[q].link
                qc.firstpos = self.states[q].firstpos
                self.states.append(qc)

                # redirecționăm tranzițiile care mergeau spre q
                while p >= 0 and self.states[p].next.get(c) == q:
                    self.states[p].next[c] = clone
                    p = self.states[p].link

                self.states[q].link = st.link = clone
        # gata extensia


def find_matches(sa: SuffixAutomaton, t: str, min_len=1):
    matches = []
    v = 0           # stare curentă
    l = 0           # lungimea curentă a potrivirii
    for i, c in enumerate(t):
        # dacă nu există tranziție, „urcăm” prin link-uri
        while v and c not in sa.states[v].next:
            v = sa.states[v].link
            l = sa.states[v].len
        if c in sa.states[v].next:
            v = sa.states[v].next[c]
            l += 1
        else:
            v = 0
            l = 0

        # dacă lungimea curentă scade sub min_len, reportăm segmentul anterior
        # (în momentul în care începem o „cădere” de lungime).
        if l < min_len or i == len(t)-1:
            # poziția finală a segmentului este i-1 dacă l<min_len,
            # altfel e i (ultimul caracter)
            end = i if l >= min_len and i == len(t)-1 else i-1
            length = prev_l if 'prev_l' in locals() else 0
            if length >= min_len:
                start_t = end - length + 1
                src_end = sa.states[prev_v].firstpos
                start_s = src_end - length + 1
                matches.append((start_t, length, start_s))
            prev_l = l
            prev_v = v
        else:
            # salvăm valorile vechi pentru când l scade
            prev_l, prev_v = l, v
    return matches


import xml.etree.ElementTree as ET

def write_xml(matches, susp_name, src_name, out_name):
    root = ET.Element('document', reference=susp_name)
    for susp_off, length, src_off in matches:
        ET.SubElement(root, 'feature', {
            'name': 'detected-plagiarism',
            'this_offset': str(susp_off),
            'this_length': str(length),
            'source_reference': src_name,
            'source_offset': str(src_off),
            'source_length': str(length)
        })
    tree = ET.ElementTree(root)
    tree.write(out_name, encoding='utf-8', xml_declaration=True)


def main():
    susp_file = 'suspicious-document020468.txt'
    src_file  = 'source-document020468.txt'
    out_file  = 'suspicious-document020468-source-document020468.xml'
    with open(src_file, 'r', encoding='utf-8') as f:
        src = f.read()
    with open(susp_file, 'r', encoding='utf-8') as f:
        susp = f.read()

    sa = SuffixAutomaton(src)
    # Alege min_len după cum vrei tu;  așa prinzi doar segmente >= min_len caractere
    matches = find_matches(sa, susp, min_len=50)

    write_xml(matches, susp_file, src_file, out_file)
    print(f"Generated {out_file} with {len(matches)} features.")

if __name__ == '__main__':
    main()
