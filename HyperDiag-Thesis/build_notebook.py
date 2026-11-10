"""Convert hyperdiag_notebook_src.py (percent-cell format) into HyperDiag_PreDefence.ipynb."""
import re, sys
import nbformat as nbf

src = open(sys.argv[1] if len(sys.argv) > 1 else "hyperdiag_notebook_src.py", encoding="utf-8").read()
out = sys.argv[2] if len(sys.argv) > 2 else "HyperDiag_PreDefence.ipynb"
nb = nbf.v4.new_notebook()
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
for head, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in head:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(md))
    elif body.strip():
        nb.cells.append(nbf.v4.new_code_cell(body))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out)
print(f"wrote {out} with {len(nb.cells)} cells")
