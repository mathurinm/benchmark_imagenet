from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torchvision.models as models
    import torch
    from benchmark_utils.accuracy import evaluate_acc_and_loss
    from benchmark_utils.checkpoint import save_checkpoint
    from benchmark_utils.model.simple_vit import (
        simple_vit_s16_in1k_butterfly,
        simple_vit_b16_in1k_butterfly,
        simple_vit_s16_in1k,
        simple_vit_b16_in1k,
    )
    from pathlib import Path
    from benchmark_utils.checkpoint import default_directory_checkpoint


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "Cross Entropy"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        "arch": ["simple_vit_s16_in1k"],
        "pretrained": [False],
        "butterfly": [False],
        "num_debfly_layer": [24],
        "debfly_version": ["densification"],
        "chain_type": ["monarch"],
        "monarch_blocks": [4],
        "num_debfly_factors": [2],
        "debfly_rank": [1],
        "chain_idx": [0],
        "batch_size": [128],
        "workers": [4],
        "criterion": ["cross_entropy"],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, trainset, valset, testset):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

        self.trainset = trainset
        self.valset = valset
        self.testset = testset

        val_sampler = None
        test_sampler = None

        # define only val and test loaders since they are used at the end of each epoch
        # to compute the validation and test metrics
        # train loader is defined in the solver as it is only used to train the model
        self.val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.batch_size,
            shuffle=(val_sampler is None),
            num_workers=self.workers,
            pin_memory=True,
            sampler=val_sampler,
        )
        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=(test_sampler is None),
            num_workers=self.workers,
            pin_memory=True,
            sampler=test_sampler,
        )

    def compute(self, solver_state):
        # print("compute")
        # print(x)
        # model, optimizer, scheduler, epoch, best_val_top_1 = x

        model = solver_state["model"]
        optimizer = solver_state["optimizer"]
        scheduler = solver_state["scheduler"]
        epoch = solver_state["epoch"]
        best_val_top1 = solver_state["best_val_top1"]

        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.

        dataloader = self.val_loader
        if self.criterion == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss not implemented")
        gpu = None
        print_freq = 10
        distributed = False
        world_size = None
        workers = self.workers
        batch_size = self.batch_size
        prefix_print = "val"

        val_top1, val_top5, val_loss = evaluate_acc_and_loss(
            dataloader,
            model,
            criterion,
            gpu,
            print_freq,
            distributed,
            world_size,
            workers,
            batch_size,
            prefix_print,
        )

        # Save checkpoint here
        # TODO: only do this on rank 0 when multi-GPU
        is_best = val_top1 > best_val_top1
        solver_state["best_val_top1"] = max(val_top1, best_val_top1)
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_val_top_1": solver_state["best_val_top1"],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
            is_best,
            directory=default_directory_checkpoint,
        )
        # TODO exp_dir where to define?

        # log
        if solver_state["logger"] is not None:
            metrics_dict = {
                "train/loss": solver_state["train_loss"],
                "train/acc1": solver_state["train_top1"]
                if isinstance(solver_state["train_top1"], float)
                or isinstance(solver_state["train_top1"], int)
                else solver_state["train_top1"].item(),
                "train/acc5": solver_state["train_top5"]
                if isinstance(solver_state["train_top5"], float)
                or isinstance(solver_state["train_top5"], int)
                else solver_state["train_top5"].item(),
                "epoch": solver_state["epoch"],
                "batch_size": self.batch_size,
                "lr": self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.lr,
                "memory": torch.cuda.max_memory_allocated() // (1024 * 1024),
                "weight_decay": self.weight_decay,
            }
            print(metrics_dict)
            solver_state["logger"].log_step(metrics_dict, solver_state["epoch"])

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=val_loss
            if isinstance(val_loss, float) or isinstance(val_loss, int)
            else val_loss.item(),
            val_loss=val_loss
            if isinstance(val_loss, float) or isinstance(val_loss, int)
            else val_loss.item(),
            val_top1=val_top1
            if isinstance(val_top1, float) or isinstance(val_top1, int)
            else val_top1.item(),
            val_top5=val_top5
            if isinstance(val_top5, float) or isinstance(val_top5, int)
            else val_top5.item(),
            # train_top1 = train_top1,
            # train_top5 = train_top5,
            # train_loss = train_loss,
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        solver_state = {
            "model": self.get_model(),
            "optimizer": None,
            "scheduler": None,
            "epoch": None,
            "best_top1_val": None,
        }
        return solver_state
        # return self.get_model()

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        model = self.get_model()

        return dict(
            model=model,
            trainset=self.trainset,
        )

    def get_model(self):
        # create model
        if self.pretrained:
            print("=> using pre-trained model '{}'".format(self.arch))
            assert not self.butterfly
            model = models.__dict__[self.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(self.arch))
            if self.butterfly:
                assert self.arch in ["simple_vit_s16_in1k", "simple_vit_b16_in1k"]
                print("with butterfly structure")
                model = eval(f"{self.arch}_butterfly")(
                    num_debfly_layer=self.num_debfly_layer,
                    version=self.debfly_version,
                    chain_type=self.chain_type,
                    monarch_blocks=self.monarch_blocks,
                    num_debfly_factors=self.num_debfly_factors,
                    rank=self.debfly_rank,
                    chain_idx=self.chain_idx,
                )
            else:
                if self.arch in ["simple_vit_s16_in1k", "simple_vit_b16_in1k"]:
                    model = eval(self.arch)()
                else:
                    model = models.__dict__[self.arch]()
        return model
