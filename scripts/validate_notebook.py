from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "GRAMM_MAXSR_V2_LOCAL_2m_to_1m.ipynb"

if not NB.exists():
    raise SystemExit(f"Notebook not found: {NB}")

nb = nbformat.read(NB, as_version=4)
checked = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    compile(cell.source, f"cell_{i}", "exec")
    checked += 1

print(f"PASS: compiled {checked} code cells")
print(NB)
