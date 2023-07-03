from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torchvision.models as models
    import torch
    from benchmark_utils.accuracy import evaluate_acc_and_loss
    from benchmark_utils.model.simple_vit import (
        simple_vit_s16_in1k_butterfly,
        simple_vit_b16_in1k_butterfly,
        simple_vit_s16_in1k,
        simple_vit_b16_in1k,
    )


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "ImageNet"

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

    def compute(self, model):
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

        top1, top5, loss = evaluate_acc_and_loss(
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

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=loss
            if isinstance(loss, float) or isinstance(loss, int)
            else loss.item(),
            val_loss=loss
            if isinstance(loss, float) or isinstance(loss, int)
            else loss.item(),
            val_top1=top1
            if isinstance(top1, float) or isinstance(top1, int)
            else top1.item(),
            val_top5=top5
            if isinstance(top5, float) or isinstance(top5, int)
            else top5.item(),
            # train_top1 = train_top1,
            # train_top5 = train_top5,
            # train_loss = train_loss,
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return self.get_model()

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
