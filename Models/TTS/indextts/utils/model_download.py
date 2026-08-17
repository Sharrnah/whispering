"""Offline-only replacement for the upstream Hugging Face downloader."""


def ensure_models_available(model_dir):
    raise FileNotFoundError(
        "IndexTTS auxiliary files are incomplete in " + str(model_dir) + ". "
        "Whispering Tiger only installs the verified application-hosted archive "
        "and never downloads model files from Hugging Face at runtime."
    )
