"""
Execute notebooks/online_shoppers_analysis.ipynb and save outputs.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "online_shoppers_analysis.ipynb"

print(f"Reading {NOTEBOOK_PATH}...")
nb = nbf.read(NOTEBOOK_PATH, as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}}
)

print("Executing notebook cells...")
client.execute()

print(f"Saving executed notebook to {NOTEBOOK_PATH}...")
nbf.write(nb, NOTEBOOK_PATH)
print("Notebook executed and saved successfully!")
