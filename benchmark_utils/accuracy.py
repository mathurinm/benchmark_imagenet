from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch
    import time
    from torch.utils.data import Subset
    AverageMeter = import_ctx.get('AverageMeter', from_='utils.meters')
    ProgressMeter = import_ctx.get('ProgressMeter', from_='utils.meters')
    Summary = import_ctx.get('Summary', from_='utils.meters')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_acc_and_loss(dataloader, model, criterion, gpu, print_freq, distributed, world_size, workers, batch_size, prefix_print):
        '''
        Evaluate accuracy and loss of the model on the given dataloader
        prefix_print: str for logging
        '''
        def run_evaluate(loader, base_progress=0):
            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if gpu is not None and torch.cuda.is_available():
                        images = images.cuda(gpu, non_blocking=True)
                    # if torch.backends.mps.is_available():
                    #     images = images.to('mps')
                    #     target = target.to('mps')
                    if torch.cuda.is_available():
                        target = target.cuda(gpu, non_blocking=True)

                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % print_freq == 0:
                        progress.display(i + 1)

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(dataloader) + (distributed and (len(dataloader.sampler)
                                                    * world_size < len(dataloader.dataset))),
            [batch_time, losses, top1, top5],
            prefix=f'{prefix_print}: ')

        # switch to evaluate mode
        model.eval()

        run_evaluate(dataloader)
        if distributed:
            top1.all_reduce()
            top5.all_reduce()

        if distributed and (len(dataloader.sampler) * world_size < len(dataloader.dataset)):
            aux_val_dataset = Subset(dataloader.dataset,
                                    range(len(dataloader.sampler) * world_size, len(dataloader.dataset)))
            aux_dataloader = torch.utils.data.DataLoader(
                aux_val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            run_evaluate(aux_dataloader, len(dataloader))

        progress.display_summary()

        return top1.avg, top5.avg, losses.avg