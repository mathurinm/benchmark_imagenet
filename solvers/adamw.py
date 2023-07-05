from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.lr_schedulers import (
        scheduler_linear_warmup_and_cosine,
        scheduler_linear_warmup_and_multistep,
    )
    from benchmark_utils.accuracy import accuracy
    from benchmark_utils.meters import AverageMeter, ProgressMeter
    from benchmark_utils.mixup import RandomMixup
    import time
    import torch
    from torch.utils.data.dataloader import default_collate
    import os
    from benchmark_utils.checkpoint import default_directory_checkpoint
    from benchmark_utils.logger import Logger
    import torchvision.models as models
    from benchmark_utils.model.simple_vit import (
        simple_vit_s16_in1k_butterfly,
        simple_vit_b16_in1k_butterfly,
        simple_vit_s16_in1k,
        simple_vit_b16_in1k,
    )

X_LR_DECAY_EPOCH = [30 / 90, 60 / 90, 80 / 90]


def train_single_epoch(
    model,
    train_loader,
    epoch,
    device,
    criterion,
    optimizer,
    scaler,
    clip_grad_norm,
    scheduler,
    print_freq,
    channels_last,
    mixup_alpha,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    # data_to_device_time = AverageMeter('DataToDevice', ':6.3f')
    losses = AverageMeter("Loss", ":.4e")
    lr = AverageMeter("Lr", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    end = time.time()

    new_train_loader = train_loader

    progress = ProgressMeter(
        len(new_train_loader),
        [batch_time, data_time, losses, lr, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    for i, (images, target) in enumerate(new_train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        end = time.time()

        # move data to the same device as model
        if channels_last:
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if mixup_alpha is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        losses.update(loss.detach().item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()

        # Scheduler step at each training iterations
        if scheduler is not None:
            lr.update(scheduler.get_last_lr()[0])
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "adamw"
    stopping_strategy = "callback"

    stopping_criterion = SufficientProgressCriterion(
        patience=60, strategy="callback"
    )
    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
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
        "lr": [0.001],
        "weight_decay": [0.0001],
        "lr_scheduler": ["cosine"],
        "max_epochs": [10],
        "warmup_percentage": [5 / 90],
        "distributed": [False],
        "workers": [4],
        "mixup_alpha": [0.2],
        "clip_grad_norm": [1],
        "channels_last": [True],
        "amp": [True],
        "gpu": [None],
        "device_ids": [None],
        "print_freq": [10],
        "resume_from_path": ["./checkpoints/model_best.pth.tar"],  # [None]
        "tensorboard": [False],
        "rank": [0],
        "log": [True],
    }

    def path_from_keys(self, *keys):
        path = default_directory_checkpoint
        for key in keys:
            if key == "warmup_percentage":
                path = path / f"{key}={getattr(self, key):.3f}"
            else:
                path = path / f"{key}={getattr(self, key)}"
        return path

    def set_saving_path(self):
        if self.butterfly:
            if self.chain_type == "monarch":
                self.saving_path = self.path_from_keys(
                    "arch",
                    "butterfly",
                    "num_debfly_layer",
                    "debfly_version",
                    "chain_type",
                    "monarch_blocks",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "lr_scheduler",
                    "max_epochs",
                    "warmup_percentage",
                    "mixup_alpha",
                    "clip_grad_norm",
                    "amp",
                )
            elif self.chain_type == "low_rank":
                self.saving_path = self.path_from_keys(
                    "arch",
                    "butterfly",
                    "num_debfly_layer",
                    "debfly_version",
                    "chain_type",
                    "debfly_rank",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "lr_scheduler",
                    "max_epochs",
                    "warmup_percentage",
                    "mixup_alpha",
                    "clip_grad_norm",
                    "amp",
                )
            elif self.chain_type == "monotone-min-params":
                self.saving_path = self.path_from_keys(
                    "arch",
                    "butterfly",
                    "num_debfly_layer",
                    "debfly_version",
                    "chain_type",
                    "num_debfly_factors",
                    "debfly_rank",
                    "chain_idx",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "lr_scheduler",
                    "max_epochs",
                    "warmup_percentage",
                    "mixup_alpha",
                    "clip_grad_norm",
                    "amp",
                )
        else:
            self.saving_path = self.path_from_keys(
                "arch",
                "batch_size",
                "lr",
                "weight_decay",
                "lr_scheduler",
                "max_epochs",
                "warmup_percentage",
                "mixup_alpha",
                "clip_grad_norm",
                "amp",
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
                assert self.arch in [
                    "simple_vit_s16_in1k",
                    "simple_vit_b16_in1k",
                ]
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

    def device_and_distributed_init_model(self):
        if not torch.cuda.is_available():
            print("using CPU, this will be slow")
        elif self.distributed:
            # For multiprocessing distributed, DistributedDataParallel
            # constructor should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if torch.cuda.is_available():
                print("Using DDP")
                if self.gpu is not None:
                    torch.cuda.set_device(self.gpu)
                    self.model.cuda(self.gpu)
                    # When using a single GPU per process and per
                    # DistributedDataParallel, we need to divide the batch size
                    # ourselves based on the total number of GPUs of the
                    # current node.
                    # args.batch_size = int(args.batch_size / ngpus_per_node)
                    # args.workers = int((args.workers + ngpus_per_node - 1)
                    # / ngpus_per_node)
                    assert self.batch_size % self.world_size == 0
                    self.batch_size = self.batch_size // self.world_size
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model, device_ids=[self.gpu]
                    )
                else:
                    self.model.cuda()
                    # DistributedDataParallel will divide and allocate
                    # batch_size to all # available GPUs if device_ids are
                    # not set
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model
                    )
        elif self.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)
            self.model = self.model.cuda(self.gpu)
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     model = model.to(device)
        else:
            # DataParallel will divide and allocate batch_size to all available
            # GPUs
            if self.device_ids is not None:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=self.device_ids
                ).cuda()
            else:
                self.model = torch.nn.DataParallel(self.model).cuda()

    def set_train_loader(self, trainset):
        # train sampler
        if self.distributed:
            raise NotImplementedError
        else:
            train_sampler = None

        # mixup
        if self.mixup_alpha is not None:
            num_classes = 1000
            mixup = RandomMixup(num_classes, p=1.0, alpha=self.mixup_alpha)

            def collate_fn(batch):
                return mixup(*default_collate(batch))

        else:
            collate_fn = None

        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )

    def set_scheduler(self):
        total_num_iterations = self.max_epochs * len(self.train_loader)
        warmup_iterations = int(self.warmup_percentage * total_num_iterations)
        if self.lr_scheduler == "constant":
            print("=>constant lr")
            self.scheduler = None
        elif self.lr_scheduler == "cosine":
            print("=>cosine")
            self.scheduler = scheduler_linear_warmup_and_cosine(
                self.optimizer,
                initial_lr=self.lr,
                warmup_iterations=warmup_iterations,
                max_iterations=total_num_iterations,
            )
            print(f"scheduler warmup iterations: {warmup_iterations}")
            print(f"total_num_iterations: {total_num_iterations}")
        elif self.lr_scheduler == "multi-step":
            print("=>multi-step")
            milestones_in_iterations = [
                int(x * total_num_iterations) for x in X_LR_DECAY_EPOCH
            ]
            self.scheduler = scheduler_linear_warmup_and_multistep(
                self.optimizer,
                gamma=0.1,
                warmup_iterations=warmup_iterations,
                milestones_in_iterations=milestones_in_iterations,
            )
            print(f"scheduler warmup iterations: {warmup_iterations}")
            print(
                "scheduler milestones in iteration number:"
                + f" {milestones_in_iterations}"
            )
        else:
            raise NotImplementedError

    def set_logger(self):
        self.logger = None
        if self.log:
            metrics_name = [
                "train/loss",
                "train/acc1",
                "train/acc5",
                "epoch",
                "batch_size",
                "weight_decay",
                "lr",
                "memory",
                "seed",
                "dp_epsilon",
            ]
            csv_dir = self.saving_path / f"rank={self.rank}" / "csv"
            if self.tensorboard:
                tensorboard_dir = (
                    self.saving_path / f"rank={self.rank}" / "tensorboard"
                )
            else:
                tensorboard_dir = None
            self.logger = Logger(
                metrics_name, csv_dir=csv_dir, tensorboard_dir=tensorboard_dir
            )

    def set_objective(self, trainset):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.model = self.get_model()

        self.device_and_distributed_init_model()
        # set_objective is launched once per solver's parameter combination
        # while run is launched several times for a given solver's parameter
        # combination so we define what is common to all runs (trainloader...)
        # here to avoid an overhead at each run

        # trainloader

        self.set_train_loader(trainset)

        # device
        if torch.cuda.is_available():
            if self.gpu:
                self.device = torch.device("cuda:{}".format(self.gpu))
            else:
                self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # loss function (criterion)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # optimizer
        parameters = self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )

        # amp
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # channels_last
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

        # scheduler
        self.set_scheduler()

        # set saving path for checkpoints and logger

        self.set_saving_path()

        # create logger for saving stats in csv
        # logger to save in csv

        self.set_logger()

        # optionally resume from a checkpoint in run()

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `max_epochs`

        # something that carries everything we want to save or track (logging
        # in csv is done in compute() of objective and not in run() of solver
        # to avoid time measurements of logging)
        solver_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "epoch": 0,
            "best_val_top1": 0,
            "logger": self.logger,
            "train_top1": None,
            "train_top5": None,
            "train_loss": None,
            "batch_size": self.batch_size,
            "saving_path": self.saving_path,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        # optionally resume from a checkpoint
        if self.resume_from_path:
            if os.path.isfile(self.resume_from_path):
                print(
                    "=> loading checkpoint '{}'".format(self.resume_from_path)
                )
                if self.gpu is None:
                    checkpoint = torch.load(self.resume_from_path)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.gpu)
                    checkpoint = torch.load(
                        self.resume_from_path, map_location=loc
                    )

                    # best_val_top_1 may be from a checkpoint from a different
                    # GPU
                solver_state["model"].load_state_dict(checkpoint["state_dict"])
                solver_state["optimizer"].load_state_dict(
                    checkpoint["optimizer"]
                )
                if solver_state["scheduler"] is not None:
                    solver_state["scheduler"].load_state_dict(
                        checkpoint["scheduler"]
                    )
                solver_state["epoch"] = checkpoint["epoch"]
                if self.gpu is not None:
                    solver_state["best_val_top1"] = checkpoint[
                        "best_val_top1"
                    ].to(self.gpu)
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        self.resume_from_path, checkpoint["epoch"]
                    )
                )
            else:
                print(
                    "=> no checkpoint found at '{}'".format(
                        self.resume_from_path
                    )
                )

        # max_epochs
        callback.stopping_criterion.max_runs = (
            self.max_epochs - solver_state["epoch"]
        )

        while callback(solver_state):
            # if self.distributed:
            #     raise NotImplementedError
            #     solver_state["train_sampler"].set_epoch(
            # solver_state["epoch"]
            # )

            # train for one epoch
            (
                solver_state["train_top1"],
                solver_state["train_top5"],
                solver_state["train_loss"],
            ) = train_single_epoch(
                solver_state["model"],
                self.train_loader,
                solver_state["epoch"],
                self.device,
                self.criterion,
                solver_state["optimizer"],
                self.scaler,
                self.clip_grad_norm,
                solver_state["scheduler"],
                self.print_freq,
                self.channels_last,
                self.mixup_alpha,
            )
            solver_state["epoch"] += 1
            # best val top 1 is updated in the compute function of the
            # objective

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        # ONLY USED FOR TESTS
        solver_state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "epoch": 0,
            "best_val_top1": 0,
            "logger": None,
            "train_top1": None,
            "train_top5": None,
            "train_loss": None,
            "batch_size": self.batch_size,
            "saving_path": self.saving_path,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        return solver_state
