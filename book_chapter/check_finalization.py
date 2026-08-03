"""Finalization checks for the book chapter.

Verifies there are no orphan citations in either direction (every \\cite key has
a bib entry and vice versa) and reports any remaining \\FILL placeholders.
Exit code 0 only if there are no orphans AND no \\FILL placeholders remain.
"""
import re
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "Lunar_Research_Book_Chapter")
tex = open(os.path.join(BASE, "main.tex")).read()
bib = open(os.path.join(BASE, "references.bib")).read()

cited = set()
for m in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
    for k in m.split(","):
        cited.add(k.strip())
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

orphan_bib = defined - cited          # bib entries never cited
orphan_cite = cited - defined         # citations with no bib entry
fills = re.findall(r"\\FILL\{", tex)

print("Uncited bib entries :", sorted(orphan_bib) or "none")
print("Undefined citations :", sorted(orphan_cite) or "none")
print("Remaining \\FILL slots:", len(fills))

ok = not orphan_bib and not orphan_cite and not fills
if not ok:
    print("\nFAIL: resolve the items above before submission.")
sys.exit(0 if ok else 1)
