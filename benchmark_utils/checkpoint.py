import torch
import shutil


def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / filename)
    if is_best:
        shutil.copyfile(directory / filename, directory / "model_best.pth.tar")
