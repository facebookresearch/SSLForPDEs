# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Callable
import torch
import numpy as np


# ========================================================
#       Generic Lie Trotter implementation
# ========================================================

def lie_trotter_exp_2(
        state: Tuple,
        functions: List[Callable],
        strengths: List[float],
        T: int = 1,
        factor: float = 1.,
        **f_args
        ):

    for _ in range(T):
        for h, t in zip(reversed(functions), reversed(strengths)):
            term = factor * t / T / 2
            state = h(term, *state, **f_args)

        for h, t in zip(functions, strengths):
            term = factor * t / T / 2
            state = h(term, *state, **f_args)

    return state


def lie_trotter_exp(
        state: Tuple,
        functions: List[Callable],
        strengths: List[float],
        order: int = 2,
        T: int = 1,
        factor: float = 1.,
        **f_args
        ):
    if T == 0:
        return state
    factor = factor / T

    for _ in range(T):
        if order == 2:
            state = lie_trotter_exp_2(state, functions, strengths, T=1, factor=factor, **f_args)
        elif order > 2:
            u_k = 1 / (4 - 4**(1 / (2 * order - 1)))
            state = lie_trotter_exp(state, functions, strengths, order=order - 2, T=1, factor=factor * u_k, **f_args)
            state = lie_trotter_exp(state, functions, strengths, order=order - 2, T=1, factor=factor * u_k, **f_args)
            state = lie_trotter_exp(state, functions, strengths, order=order - 2, T=1, factor=factor * (1 - 4 * u_k), **f_args)
            state = lie_trotter_exp(state, functions, strengths, order=order - 2, T=1, factor=factor * u_k, **f_args)
            state = lie_trotter_exp(state, functions, strengths, order=order - 2, T=1, factor=factor * u_k, **f_args)
        elif order == 0:
            pass
        else:
            raise NotImplementedError()
    return state



# ========================================================
#        Symmetry related functions for Navier-Stokes
# ========================================================

class NSTransforms:
    # time translation
    @staticmethod
    def group_1(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t + g, x, y, u, v
        else:
            return t + g, x, y, u, v, px, py

    # x translation
    @staticmethod
    def group_2(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x + g, y, u, v
        else:
            return t, x + g, y, u, v, px, py

    # y translation
    @staticmethod
    def group_3(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x, y + g, u, v
        else:
            return t, x, y + g, u, v, px, py

    # scale change
    @staticmethod
    def group_4(g, t, x, y, u, v, px=None, py=None):
        g = torch.exp(g)
        if px is None:
            return g * g * t, g * x, g * y, u / g, v / g
        else:
            return g * g * t, g * x, g * y, u / g, v / g, px / (g * g), py / (g * g)

    # rotation
    @staticmethod
    def group_5(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, torch.cos(g) * x - torch.sin(g) * y, torch.sin(g) * x + np.cos(g) * y, torch.cos(g) * u - torch.sin(g) * v, torch.sin(g) * u + torch.cos(g) * v
        else:
            return t, torch.cos(g) * x - torch.sin(g) * y, torch.sin(g) * x + np.cos(g) * y, torch.cos(g) * u - torch.sin(g) * v, torch.sin(g) * u + torch.cos(g) * v, px, py

    # Linear E(x)
    @staticmethod
    def group_6(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x + g * t, y, u + g, v
        else:
            return t, x + g * t, y, u + g, v, px, py

    # Linear E(y)
    @staticmethod
    def group_7(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x, y + g * t, u, v + g
        else:
            return t, x, y + g * t, u, v + g, px, py

    # Quadratic E(x)
    @staticmethod
    def group_8(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x + g * t, y, u + g, v
        else:
            return t, x + g * t * t, y, u + 2 * g * t, v, px - g, py

    # Quadratic E(y)
    @staticmethod
    def group_9(g, t, x, y, u, v, px=None, py=None):
        if px is None:
            return t, x, y + g * t * t, u, v + g
        else:
            return t, x, y + g * t, u, v + 2 * g * t, px, py - g

    def apply(self, gs, t, x, y, u, v, px=None, py=None, order=4, steps=1, **f_args):
        group_ops = [
                    NSTransforms.group_1,
                    NSTransforms.group_2,
                    NSTransforms.group_3,
                    NSTransforms.group_4,
                    NSTransforms.group_5,
                    NSTransforms.group_6,
                    NSTransforms.group_7,
                    NSTransforms.group_8,
                    NSTransforms.group_9,
                ]

        if px is None:
            state = lie_trotter_exp(
                (t, x, y, u, v),
                group_ops,
                gs,
                order=order,
                T=steps,
                **f_args
                )
            return state[0], state[1], state[2], state[3], state[4]
        else:
            state = lie_trotter_exp(
                (t, x, y, u, v, px, py),
                group_ops,
                gs,
                order=order,
                T=steps,
                **f_args
                )
            return state[0], state[1], state[2], state[3], state[4], state[5], state[6]

