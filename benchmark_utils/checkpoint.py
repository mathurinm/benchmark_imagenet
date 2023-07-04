import torch
import shutil


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, dir / filename)
    if is_best:
        shutil.copyfile(dir / filename, dir / 'model_best.pth.tar')