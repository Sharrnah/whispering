"""Populate the standalone app's NLTK resources during the Docker build."""
from pathlib import Path
import nltk

destination = Path(".cache/nltk").resolve()
for package in ("punkt", "punkt_tab", "wordnet"):
    if not nltk.download(package, download_dir=str(destination), raise_on_error=True):
        raise RuntimeError(f"Could not prepare NLTK resource: {package}")
nltk.data.path.insert(0, str(destination))
assert nltk.sent_tokenize("One sentence. Another sentence.") == ["One sentence.", "Another sentence."]
