# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import h5py
import json
import numpy as np
import time
import torch
import torchvision

from pathlib import Path
from torch import nn
from typing import Optional, List, Tuple
from torch.utils.data import Dataset


from transformations import NSTransforms
# ========================================================
#           Datasets and data-loader Navier-Stokes
# ========================================================


class RandomCrop3d(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, tensor):
        C, T, H, W = tensor.size()
        t, h, w = self.crop_size

        if t > T or h > H or w > W:
            raise ValueError("Crop size must be smaller than input size")

        left = torch.randint(0, W - w + 1, size=(1,))
        top = torch.randint(0, H - h + 1, size=(1,))
        start = torch.randint(0, T - t + 1, size=(1,))

        right = left + w
        bottom = top + h
        end = start + t

        return tensor[..., start:end, top:bottom, left:right]



# This class is used as our torch augmentations where we sample transformations and apply them
class LPSNavierStokes(object):
    def __init__(
              self,
              transforms_strength: Optional[List[float]] = [0] * 9,
              steps: Optional[int] = 2,
              order: Optional[int] = 2,
              crop_size: Optional[Tuple[int]] = (56, 128, 128),
              ) -> None:

        self.transforms_strength = transforms_strength
        self.crop = RandomCrop3d(crop_size)
        self.steps = steps
        self.order = order

    def __call__(
            self,
            sample: torch.Tensor,
            ) -> torch.Tensor:

        x, y, t, vx, vy = sample

        lie_transform = NSTransforms()

        vals = []
        vals.append(np.random.uniform(0, self.transforms_strength[0]))
        for strength in self.transforms_strength[1:]:
            vals.append(np.random.uniform(-strength, strength))

        if self.steps == 0:
            t_2, x_2, y_2, vx_2, vy_2 = t,x,y,vx,vy
        else:
            t_2, x_2, y_2, vx_2, vy_2 = lie_transform.apply(
                torch.tensor(vals),
                t,
                x,
                y,
                vx,
                vy,
                order=self.order,
                steps=self.steps,
                )
        image = torch.stack((x_2, y_2, t_2, vx_2, vy_2)).to(torch.float32)
        image = self.crop(image)

        return image


class NSDataset(Dataset):
    def __init__(
            self,
            data_root: str = "/data/pde_arena/NavierStokes2D_cond_smoke_v1/",
            transforms_strength: Optional[List[float]] = [0] * 9,
            steps: Optional[int] = 2,
            order: Optional[int] = 2,
            mode:  Optional[str] = "train",
            crop_size: Optional[Tuple[int]] = (56, 64, 64),
            size: Optional[int] = 100000,
            ):

        self.data_root = Path(data_root)
        self.mode = mode
        self.size=size

        self.transform = LPSNavierStokes(
            transforms_strength=transforms_strength,
            steps=steps,
            order=order,
            crop_size=crop_size,
            )

        self.h5_files = []

        #Get all training files
        files = []
        for file in self.data_root.iterdir():
            if str(file)[-2:] != "h5" or not self.mode in str(file):
                continue
            files.append(str(file))

        #Sort files
        def extract_seed(path):
            path = str(path)
            if "train" in path:
                seed = int(path.split('_')[-3])
            else:
                seed = int(path.split('_')[-2])
            return seed    
        files.sort(key=extract_seed)

        #Create data
        for file in files:
            self.h5_files.append(h5py.File(str(file), 'r'))

        self.size_per_file = 32

    def __len__(self):
        return len(self.h5_files * self.size_per_file)

    def __getitem__(self, idx):
        idx = idx % self.size

        file = idx // self.size_per_file
        index = idx % self.size_per_file

        if self.mode == "val":    
            data = self.h5_files[file]['valid']
        else:
            data = self.h5_files[file][self.mode]

        vx = torch.Tensor(data['vx'][index])
        vy = torch.Tensor(data['vy'][index])
        x_ = data['x'][index]
        y_ = data['y'][index]
        t_ = data['t'][index]

        x = torch.Tensor(np.tile(np.tile(x_, (y_.shape[0], 1)), (t_.shape[0], 1, 1)))
        y = torch.Tensor(np.tile(np.tile(y_, (x_.shape[0], 1)).T, (t_.shape[0], 1, 1)))
        t = torch.Tensor(np.tile(t_, (x_.shape[0], y_.shape[0], 1)).T)

        b = data['buo_y'][index]

        sample = (x, y, t, vx, vy) 
        view_1 = self.transform(sample)
        view_2 = self.transform(sample)
        view_1 = view_1.flatten(0, 1)
        view_2 = view_2.flatten(0, 1)

        return view_1, view_2, np.float32(b)


def get_loader_ns(data_root, batch_size, steps, order, num_workers, strengths, mode, crop_size, dataset_size):
    dataset = NSDataset(data_root, strengths, steps, order, mode=mode, crop_size=crop_size,size=dataset_size)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader


class NSDatasetEval(Dataset):
    def __init__(
            self,
            data_root: str = "/data/pde_arena/NavierStokes2D_cond_smoke_v1/",
            mode: str = "test",
            crop_size: Optional[Tuple[int]] = (56, 64, 64)
            ):

        self.data_root = Path(data_root)
        self.mode = mode

        self.transform = LPSNavierStokes(
            crop_size=crop_size,
            order=0,
            steps=0,
            ) 
    
        self.h5_files = []
        
        #Get all training files
        files = []
        for file in self.data_root.iterdir():
            if str(file)[-2:] != "h5" or not self.mode in str(file):
                continue
            files.append(str(file))
        
        #Sort files
        def extract_seed(path):
            path = str(path)
            if "train" in path:
                seed = int(path.split('_')[-3])
            else:
                seed = int(path.split('_')[-2])
            return seed    
        files.sort(key=extract_seed)
        
        #Create data
        for file in files:
            self.h5_files.append(h5py.File(str(file),'r'))
     
        self.size_per_file = 32
        
    def __len__(self):
        return len(self.h5_files*self.size_per_file)

    def __getitem__(self, idx):
        file = idx // self.size_per_file
        index = idx % self.size_per_file
        
        if self.mode == "val":    
            data = self.h5_files[file]['valid']
        else:
            data = self.h5_files[file][self.mode]
        
        vx = torch.Tensor(data['vx'][index])
        vy = torch.Tensor(data['vy'][index])
        x_ = data['x'][index]
        y_ = data['y'][index]
        t_ = data['t'][index]

        x =  torch.Tensor(np.tile(np.tile(x_,(y_.shape[0],1)),(t_.shape[0],1,1)))
        y = torch.Tensor(np.tile(np.tile(y_,(x_.shape[0],1)).T,(t_.shape[0],1,1)))
        t = torch.Tensor(np.tile(t_,(x_.shape[0],y_.shape[0],1)).T)

        b = data['buo_y'][index]
        
        sample = (x, y, t, vx, vy) 
        view_1 = self.transform(sample)
        view_1 = view_1.flatten(0, 1)
        return view_1, np.float32(b)


def get_eval_loader_ns(data_root, batch_size, num_workers, mode,crop_size):
    dataset = NSDatasetEval(data_root,mode=mode,crop_size=crop_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader

# ========================================================
#                Random utils functions
# ========================================================

def log_stats(stats, writer, epoch):
    for k, v in stats.items():
        writer.add_scalar(k, v, epoch)


def log_imgs(imgs_to_log, writer, epoch):
    legend, imgs = imgs_to_log
    imgs = [torch.unsqueeze(img.detach(), dim=0) for img in imgs]
    grid = torchvision.utils.make_grid(imgs, nrow=1, normalize=True, scale_each=True, padding=2, pad_value=1.0)
    writer.add_image(legend, grid, epoch)


class RangeSigmoid(nn.Module):
    def __init__(self, max, min):
        super().__init__()
        self.max = max
        self.min = min

    def forward(self, input):
        return torch.sigmoid(input) * (self.max - self.min) + self.min


def log(folder, content, start_time):
        print(f'=> Log: {content}')
        # if self.rank != 0: return
        cur_time = time.time()
        with open(folder + '/log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - start_time,
                **content
            }) + '\n')
            fd.flush()


def relative_error(y_pred, y):
    err = torch.abs(y_pred - y) / torch.abs(y)
    return torch.mean(err)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
