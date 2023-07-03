from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.lr_schedulers import scheduler_linear_warmup_and_cosine, scheduler_linear_warmup_and_multistep
    from benchmark_utils.accuracy import accuracy
    from benchmark_utils.meters import AverageMeter, ProgressMeter
    from benchmark_utils.mixup import RandomMixup
    import time
    import torch
    from torch.utils.data.dataloader import default_collate

X_LR_DECAY_EPOCH = [30 / 90, 60 / 90, 80 / 90]


def train_single_epoch(model, train_loader, epoch, device, criterion, optimizer, scaler, clip_grad_norm, scheduler, print_freq, channels_last, mixup_alpha):
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
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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
    name = 'adamw'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'batch_size': [128],
        'lr': [0.01],
        'weight_decay': [0.01],
        'lr_scheduler': ['cosine'],
        'epochs': [90],
        'warmup_percentage': [0],
        'distributed': [False],
        'workers': [4],
        'mixup_alpha': [None],
        'criterion': ['cross_entropy'],
        'channel_last': [True],
        'amp': [True],
        'gpu': [None],
    }

    def set_objective(self, model, trainset):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.model = model

        # set_objective is launched once per solver's parameter combination while run is launched several times for a given solver's parameter combination so we define what is common to all runs (trainloader...) here to avoid an overhead at each run

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
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # optimizer
        parameters = self.model.parameters()
        self.optimizer = torch.optim.AdamW(parameters, lr=self.lr,
                                      weight_decay=self.weight_decay)

        # amp
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # channels_last
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

        # scheduler
        total_num_iterations = self.epochs * len(self.train_loader)
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
            print(f"scheduler milestones in iteration number: {milestones_in_iterations}")
        else:
            raise NotImplementedError

    def run(self, epochs):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `epochs`.

        for epoch in range(epochs):
            # time
            begin = time.time()

            if self.distributed:
                raise NotImplementedError
                train_sampler.set_epoch(epoch)

            # train for one epoch
            self.train_top1, self.train_top5, self.train_loss = train_single_epoch(self.model, self.train_loader, epoch, self.device, self.criterion, self.optimizer, self.scaler, self.clip_grad_norm, self.scheduler, self.print_freq, self.channels_last, self.mixup_alpha)



    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model, self.train_top1, self.train_top5, self.train_loss