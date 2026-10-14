"""Turn analysis.py (cells separated by '# %%') into analysis.ipynb and execute it top to bottom."""
import nbformat as nbf
from pathlib import Path
import sys
from jupyter_client import KernelManager
from nbconvert.preprocessors import ExecutePreprocessor

src = open("analysis.py", encoding="utf8").read()
nb = nbf.v4.new_notebook()
for chunk in src.split("# %%")[1:]:
    chunk = chunk.strip("\n")
    if chunk.startswith(" [markdown]"):
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:])
        nb.cells.append(nbf.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbf.v4.new_code_cell(chunk.strip("\n")))
# Use the interpreter running this script, so the notebook uses this project's dependencies.
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
history_dir = Path("data/interim/ipython").resolve()
history_dir.mkdir(parents=True, exist_ok=True)
km.kernel_spec.env["IPYTHONDIR"] = str(history_dir)
try:
    ExecutePreprocessor(timeout=3000, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}}, km=km)
finally:
    if km.has_kernel:
        km.shutdown_kernel(now=True)
nbf.write(nb, "analysis.ipynb")
print("analysis.ipynb executed")
