from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import shutil
    from pathlib import Path

default_directory_checkpoint = Path("./checkpoints")


def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / filename)
    if is_best:
        shutil.copyfile(directory / filename, directory / "model_best.pth.tar")
