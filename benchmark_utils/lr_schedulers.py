from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import bisect
    import torch
    import numpy as np


class WarmupMultistep:
    """iteration here is the number of training iterations (not epochs)"""

    def __init__(self, warmup_iterations, milestones_in_iterations, gamma):
        self.warmup_iterations = warmup_iterations
        self.milestones_in_iterations = sorted(milestones_in_iterations)
        self.gamma = gamma
        assert self.milestones_in_iterations[0] > warmup_iterations

    def __call__(self, iteration):
        if iteration <= self.warmup_iterations:
            factor = iteration / self.warmup_iterations
        else:
            power = bisect.bisect_right(self.milestones_in_iterations, iteration)
            factor = self.gamma**power
        return factor


def scheduler_linear_warmup_and_multistep(
    optimizer, gamma, warmup_iterations, milestones_in_iterations
):
    """This scheduler will be called at each training iteration, not at each epoch"""
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupMultistep(warmup_iterations, milestones_in_iterations, gamma),
    )
    return scheduler


class WarmupCosine:
    def __init__(self, warmup_end, max_iter, factor_min):
        self.max_iter = max_iter
        self.warmup_end = warmup_end
        self.factor_min = factor_min

    def __call__(self, iteration):
        if iteration < self.warmup_end:
            factor = iteration / self.warmup_end
        else:
            iteration = iteration - self.warmup_end
            max_iter = self.max_iter - self.warmup_end
            iteration = (iteration / max_iter) * np.pi
            factor = self.factor_min + 0.5 * (1 - self.factor_min) * (
                np.cos(iteration) + 1
            )
        return factor


def scheduler_linear_warmup_and_cosine(
    optimizer, initial_lr, warmup_iterations, max_iterations, min_lr=0.0
):
    # min_lr = 0 because: https://github.com/google-research/big_vision/blob/47ac2fd075fcb66cadc0e39bd959c78a6080070d/big_vision/utils.py#L929
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            warmup_iterations,
            max_iterations,
            min_lr / initial_lr,
        ),
    )
    return scheduler
