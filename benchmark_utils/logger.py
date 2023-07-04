from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path
    import pandas as pd
    import torch.utils.tensorboard

    # from datetime import datetime


class Logger:
    def __init__(self, metrics_name, csv_dir, tensorboard_dir=None, verbose=False):
        # now = datetime.now()
        # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.metrics_name = metrics_name
        # (csv_dir / dt_string).mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # self.csv_stats = PandasStats(csv_dir / dt_string / "results.csv", metrics_name)
        self.csv_stats = PandasStats(csv_dir / "results.csv", metrics_name)

        if tensorboard_dir is not None:
            self.writer = torch.utils.tensorboard.SummaryWriter(
                # tensorboard_dir / dt_string
                tensorboard_dir
            )
        else:
            self.writer = None
        self.verbose = verbose

    def log_step(self, metrics_dict, step):
        # Pour éviter des erreurs on rajoute la vérification des métriques names
        for k in self.metrics_name:
            assert k in self.metrics_name
        dict_for_csv = dict([(k, None) for k in self.metrics_name])
        dict_for_csv.update(metrics_dict)
        self.csv_stats.update(dict_for_csv)
        if self.writer is not None:
            for k, v in metrics_dict.items():
                if v is not None:
                    self.writer.add_scalar(k, v, step)

        if self.verbose:
            print_str = "\t".join([f"{k} {v:.4f}" for k, v in metrics_dict.items()])
            print(print_str)

        # f"Epoch: [{current_epoch}][{step}]\t"
        # f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        # f"train/loss {train_loss_meter.val:.4f} ({train_loss_meter.avg:.4f})\t"
        # f"train/loss_without_reg {loss_without_reg_val:.4f} ({loss_without_reg_avg:.4f})\t"
        # f"train/reg {reg_val:.4f} ({reg_avg:.4f})\t"
        # f"train/tau_reg {tau * reg_val:.4f} ({tau * reg_avg:.4f})\t"
        # f"train/acc {train_acc_meter.val:.1f} ({train_acc_meter.avg:.1f})\t"
        # f"Lr: {lr:.4f} \t"
        # f"Max memory: {memory_mb:.1f} MB"
        # )

    def close(self):
        if self.writer is not None:
            self.writer.close()


class PandasStats:
    def __init__(self, csv_path, columns):
        self.path = Path(csv_path)
        self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row
        if save:
            # print(self.path)
            self.stats.to_csv(self.path)

    def append(self, df, save=True):
        self.stats = self.stats.append(df)
        if save:
            # print(self.path)
            self.stats.to_csv(self.path)
