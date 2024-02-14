# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from pathlib import Path
import argparse
import os
import submitit
import time
import uuid

import subprocess
import torch
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from utils import log_stats, log, off_diagonal,get_loader_ns, get_eval_loader_ns, RangeSigmoid


from torch.utils.tensorboard import SummaryWriter


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain PDEs representations with VICReg", add_help=False
        )

    # augmentation
    parser.add_argument("--trotter-steps", type=int, default=2,
                        help='Number of steps in Trotter approximation')
    parser.add_argument("--trotter-order", type=int, default=2,
                        help='Order in Trotter approx, must be pair')

    parser.add_argument("--tsl-t", type=float, default=0.0,
                        help='Lie 1 max strength')
    parser.add_argument("--tsl-x", type=float, default=1.0,
                        help='Lie 2 max strength')
    parser.add_argument("--tsl-y", type=float, default=1.0,
                        help='Lie 3 max strength')
    parser.add_argument("--scale", type=float, default=0.1,
                        help='Lie 4 max strength')
    parser.add_argument("--rot", type=float, default=0.1,
                        help='Lie 5 max strength')
    parser.add_argument("--lin-x", type=float, default=0.01,
                        help='Lie 6 max strength')
    parser.add_argument("--lin-y", type=float, default=0.01,
                        help='Lie 7 max strength')
    parser.add_argument("--quad-x", type=float, default=0.01,
                        help='Lie 8 max strength')
    parser.add_argument("--quad-y", type=float, default=0.01,
                        help='Lie 9 max strength')

    parser.add_argument("--crop-x", type=int, default=128,
                        help='crop x')
    parser.add_argument("--crop-y", type=int, default=128,
                        help='crop y')
    parser.add_argument("--crop-t", type=int, default=16,
                        help='crop t')

    parser.add_argument("--dataset-size", type=int, default=26624,
                        help='pretraining dataset size')

    # model
    parser.add_argument("--mlp", default="512-512-512",
                        help='Size and number of layers of the MLP proj')
    parser.add_argument("--no-proj", action='store_true',
                        help='Replace the projector with identity')
    parser.add_argument("--target-dim", type=int, default=1,
                        help='Number of dimensions for regression target')

    # optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Batch size for VICReg')
    parser.add_argument("--batch-size-eval", type=int, default=128,
                        help='Batch size for the evaluation')
    parser.add_argument("--ssl-lr", type=float, default=3e-4,
                        help='LR for VICReg training')
    parser.add_argument("--eval-lr", type=float, default=0.0007,
                        help='LR for evaluation')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # loss, only playing with cov_coeff should be enough
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # data
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="/data/pde_arena/NavierStokes2D_cond_smoke_v1/")

    # logging
    parser.add_argument("--logging-folder", type=str,
                        default='/experiments/ssl_pde_NS')
    parser.add_argument("--exp-name", type=str, default=str(datetime.now()))

    # baseline
    parser.add_argument("--supervised-only", type=int, default=0,
                        help='only compute supervised baseline')



    return parser


class VICReg(nn.Module):
    def __init__(
            self,
            sim_coeff: float,
            cov_coeff: float,
            std_coeff: float,
            batch_size: int,
            mlp: str,
            n_time_steps: int,
            ):
        super().__init__()
        self.num_features = int(mlp.split("-")[-1])
        import torchvision.models.resnet as resnet
        self.backbone = resnet.__dict__['resnet18'](pretrained=False)
        self.backbone.fc = nn.Identity()
        # Change number of channels
        self.backbone.conv1 = torch.nn.Conv2d(
            n_time_steps * 5,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
            )
        _, out_dim = self.backbone(torch.zeros(1, n_time_steps*5, 224, 224)).shape

        self.projector = Projector(out_dim, mlp)
        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff

    def forward(self, x, y):
        # we do not apply inverse transform for now
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


def Projector(embedding, dimensions):
    mlp_spec = f"{embedding}-{dimensions}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def RegressionHead(embedding, dimensions, sig_max=0.1, sig_min=0.6):
    mlp_spec = f"{embedding}-{dimensions}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    # we add a sigmoid to constrain the value in a range,
    # making it easier for the classifier to be calibrated
    layers.append(RangeSigmoid(max=sig_max, min=sig_min))
    return nn.Sequential(*layers)


class ResNetBaseline(nn.Module):
    def __init__(self, target_dim, n_time_steps) -> None:
        super().__init__()
        import torchvision.models.resnet as resnet
        self.backbone = resnet.__dict__['resnet18'](pretrained=False)
        self.backbone.fc = nn.Identity()

        self.backbone.conv1 = torch.nn.Conv2d(
            n_time_steps * 5, self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias,
            )
        _, out_dim = self.backbone(torch.zeros(1, n_time_steps * 5, 224, 224)).shape
        self.regression_head = RegressionHead(out_dim, target_dim)

    def forward(self, input):
        output = self.backbone(input)
        return self.regression_head(output)


class PDETrainer():
    def __init__(self, args) -> None:
 
        self.args = args
        self.strengths = [
                args.tsl_t,
                args.tsl_x,
                args.tsl_y,
                args.scale,
                args.rot,
                args.lin_x,
                args.lin_y,
                args.quad_x,
                args.quad_y,
            ]
        self.crop_size = (args.crop_t, args.crop_x, args.crop_y)
        print(args)

    def train(self):
        root_folder = self.args.logging_folder
        logging_folder = os.path.join(self.args.logging_folder, args.exp_name)
        if not os.path.exists(logging_folder):
            os.mkdir(logging_folder)

        # This loader is used to train our SSL model
        train_loader = get_loader_ns(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            mode="train",
            crop_size=self.crop_size,
            strengths=self.strengths,
            steps=self.args.trotter_steps,
            order=self.args.trotter_order,
            num_workers=self.args.num_workers,
            dataset_size=self.args.dataset_size
            )
        # This loader is used to train the evaluation head
        eval_train_loader = get_eval_loader_ns(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            mode="val",
            crop_size=self.crop_size,
            num_workers=self.args.num_workers
            )
        # This loader is used to evaluate the evaluation head
        eval_val_loader = get_eval_loader_ns(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            mode="test",
            crop_size=self.crop_size,
            num_workers=self.args.num_workers
            )

        model = VICReg(
            sim_coeff=self.args.sim_coeff,
            cov_coeff=self.args.cov_coeff,
            std_coeff=self.args.std_coeff,
            batch_size=self.args.batch_size,
            mlp=self.args.mlp,
            n_time_steps=self.crop_size[0],
        ).cuda()

        # Only do weight decay on non-batchnorm parameters
        all_params = list(model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': self.args.wd
        }]
        optimizer = torch.optim.AdamW(param_groups, lr=self.args.ssl_lr)

        writer = SummaryWriter(
            log_dir=f"{root_folder}/runs/{self.args.exp_name}"
            )

        start_epoch = 0

        if (Path(f"{logging_folder}/model.pth")).is_file():
            print('Resuming from checkpoint')
            ckpt = torch.load(f"{logging_folder}/model.pth")
            start_epoch = ckpt['epoch'] + 1 # Since we save the model at the end of epoch
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])

        start_time = time.time()

        # random and supervised baseline
        if start_epoch == 0:
            model.eval()
            baseline_errs, stats, *imgs_to_logs = self.eval_coeff(
                                                    model.backbone,
                                                    eval_train_loader,
                                                    eval_val_loader,
                                                    self.args.target_dim,
                                                    )

            # logging
            torch.save(baseline_errs, f"{logging_folder}/baseline_errs.pt")
            log_stats(stats, writer, -1)
            log(logging_folder, stats, start_time)

        if self.args.supervised_only:
            return

        # training
        for epoch in range(start_epoch, self.args.epochs):
            model.train()
            print('epoch ' + str(epoch))

            # forward-backward
            for x, y, _ in tqdm(iter(train_loader)):
                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                ssl_loss = model.forward(x, y)
                ssl_loss.backward()
                optimizer.step()

                current_time = time.time()

            # eval
            if not(epoch % 10):
                model.eval()
                prediction_errs, stats, *imgs_to_logs = self.eval_coeff(
                                                        model.backbone,
                                                        eval_train_loader,
                                                        eval_val_loader,
                                                        self.args.target_dim,
                                                        train_supervised=False
                                                        )
                # logging
                stats = {
                    'Loss/vicreg': ssl_loss.item(),
                    'Relative time per epoch': int(current_time - start_time),
                    'Learning rate': optimizer.param_groups[0]['lr'],
                    'Epoch': epoch,
                    **stats
                }
                log_stats(stats, writer, epoch)
                log(logging_folder, stats, start_time)
                self.checkpoint(
                    epoch,
                    model,
                    optimizer,
                    f"{logging_folder}/model.pth"
                    )

        # final eval
        model.eval()
        prediction_errs, stats, *imgs_to_logs = self.eval_coeff(
                                                model.backbone,
                                                eval_train_loader,
                                                eval_val_loader,
                                                self.args.target_dim,
                                                train_supervised=False
                                                )
        # final logging
        stats = {
            'Loss/vicreg': ssl_loss.item(),
            'Relative time per epoch': int(current_time - start_time),
            'Learning rate': optimizer.param_groups[0]['lr'],
            'Epoch': epoch,
            **stats
        }
        log_stats(stats, writer, epoch)
        log(logging_folder, stats, start_time)

        torch.save(prediction_errs, f"{logging_folder}/prediction_errs.pt")
        torch.save({"model": model.state_dict()}, f"{logging_folder}/model.pth")

    def checkpoint(self, epoch, model, optimizer, logging_folder):
        state = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        torch.save(state, logging_folder)
        print("=> Saved model at: ", logging_folder, "\n")

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, args, ngpus_per_node, world_size, dist_url):
        trainer = cls(args=args) # , ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        trainer.train()

    def eval_coeff(
            self,
            backbone,
            train_loader,
            val_loader,
            target_dim,
            train_supervised=True
            ):
        print('Evaluating coefficient in NS...')

        # we may want to train the regression models online in the future
        regression_head = RegressionHead(512, target_dim).cuda()
        p_opt = torch.optim.Adam(regression_head.parameters(), lr=self.args.eval_lr)
        regression_head.train()

        # training
        if train_supervised:
            baseline = ResNetBaseline(target_dim=self.args.target_dim, n_time_steps=self.crop_size[0]).cuda()
            b_opt = torch.optim.Adam(baseline.parameters(), lr=self.args.eval_lr)
            baseline.train()

        for _ in tqdm(range(30)):
            for x, coeffs in iter(train_loader):
                # the task is kinematic viscosity prediction
                x = x.cuda()
                coeffs = coeffs.cuda()
                p_opt.zero_grad()
                # we detach to avoid gradient accumulation
                p_out = regression_head(backbone(x).detach())
                p_loss = F.mse_loss(p_out.flatten(), coeffs)
                p_loss.backward()
                p_opt.step()

                if train_supervised:
                    b_opt.zero_grad()
                    b_out = baseline(x)
                    b_loss = F.mse_loss(b_out.flatten(), coeffs)
                    b_loss.backward()
                    b_opt.step()

        p_test_loss = 0
        regression_head.eval()

        # eval
        if train_supervised:
            b_test_loss = 0
            baseline.eval()

        with torch.no_grad():
            p_errs = []
            b_errs = []
            for x, coeffs in iter(val_loader):
                x = x.cuda()
                coeffs = coeffs.cuda()
                p_test_out = regression_head(backbone(x))
                p_test_loss += F.mse_loss(p_test_out.flatten(), coeffs)
                p_errs.append([coeffs, p_test_out.detach()])

                if train_supervised:
                    b_test_out = baseline(x)
                    b_test_loss += F.mse_loss(b_test_out.flatten(), coeffs)
                    b_errs.append([coeffs, b_test_out.detach()])

        stats = {
            'Loss/prediction_eval_train': p_loss.item(),
            'Loss/prediction_eval_test': p_test_loss.item() / len(val_loader),
        }

        if train_supervised:
            stats = {
                'Loss/baseline_eval_train': b_loss.item(),
                'Loss/baseline_eval_test': b_test_loss.item() / len(val_loader),
                **stats
            }
            return b_errs, stats, ('Eval buoyancy train (Random, Baseline, GT)', [p_out.T, b_out.T, coeffs.T]), ('Eval buoyancy test (Random, Baseline, GT)', [p_test_out.T, b_test_out.T, coeffs.T])
        else:
            return p_errs, stats, ('Eval buoyancy train (SSL, GT)', [p_out.T, coeffs.T]), ('Eval buoyancy test (SSL, GT)', [p_test_out.T, coeffs.T])


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    path = "/checkpoint/"
    if Path(path).is_dir():
        p = Path(f"{path}{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args, num_gpus_per_node, dist_url):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = os.path.join(args.logging_folder, args.exp_name)
        self.dist_url = dist_url
        self.args = args

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        # MODIFIED LINE
        empty_trainer = type(self)(self.args, self.num_gpus_per_node, self.dist_url)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:58492"
        else:
            dist_url = "tcp://localhost:58492"
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        PDETrainer._exec_wrapper(gpu, self.args, self.num_gpus_per_node, world_size, dist_url)


def run_submitit(args, ngpus=1, nodes=1, timeout=4320, partition='PARTITION', comment='', use_volta32=True):
    # Path(args.logging_folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=os.path.join(args.logging_folder, args.exp_name), slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    kwargs = {}
    if use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if comment:
        kwargs['slurm_comment'] = comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.exp_name)

    dist_url = get_init_file().as_uri()

    trainer = Trainer(args, num_gpus_per_node, dist_url)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {os.path.join(args.logging_folder, args.exp_name)}")


def main(args):
    run_submitit(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'PDE VICReg training script', parents=[get_arguments()]
        )
    args = parser.parse_args()
    main(args)
