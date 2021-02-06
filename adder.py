'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math


def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding,
                                       stride=stride).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = adder.apply(W_col, X_col)

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


def uniform_quant(x, bits, special=False):
    if not special:
        min_val = x.min()
        delta = (x.max() - min_val) / (1 << bits)
    else:
        min_val = 0
        delta = 3 / (1 << bits)
    return ((x - min_val) / delta + 0.5).int() * delta + min_val


class adder(Function):
    @staticmethod
    def xor_kernel(W: torch.Tensor, X: torch.Tensor):
        # print(f"Shape W_col: {W.shape}, X_col: {X.shape}")  # W_max 70 X max 8, W_min -70 X min 0
        threshold = 2
        X = X.clamp(0, threshold)
        neg_W = W - W.relu()  # neg part
        pos_W = W.clamp(0, threshold)
        far_W = W.relu() - pos_W

        non_zero = ((0 < pos_W) & (pos_W < threshold)).int().sum()
        print(f"total: {pos_W.nelement()}, non-zero: {non_zero}, sparsity: {non_zero.float() * 100 / pos_W.nelement()}")
        # sparsity is about 33%, only 33% value is in [0, 3] -> full acc
        # sparsity is about 27%, in [0, 2] -> acc -2%
        # sparsity is about 20%, in [0, 2] -> acc -22%

        # record = []
        # for idi, i in enumerate(pos_W):
        #     for idj, j in enumerate(i):
        #         record.append((X[0, idj, :] < j).float().mean().item())
        # print(record)

        print((pos_W * 10).int()[-3, :, 0], (X * 10).int()[0, :, 123])

        far_contribution = far_W.sum(1)
        neg_contribution = -neg_W.sum(1)

        pos_W = (pos_W * (1 << 16)).int().clamp(0, threshold * (1 << 16) - 1)
        X = (X * (1 << 16)).int()
        pos_contribution = ((pos_W - X).abs().float() / (1 << 16)).sum(1)
        # pos_contribution = (pos_W.bitwise_xor(X).float() / 1024 / 1024).sum(1)
        # print(f"outshape: {pos_contribution.shape}")
        # print(pos_contribution[:5, :10])
        return -pos_contribution - neg_contribution - far_contribution

    @staticmethod
    def kernel(W, X):  # W shape of (16, 144, 1)  X shape of (1, 144, 124k)
        # W = uniform_quant(W, 8, special=True)
        # X = uniform_quant(X, 8)
        return -(W - X).abs().sum(1)

    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col, X_col)
        # output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        output = adder.kernel(W_col.unsqueeze(2), X_col.unsqueeze(0))
        # output = adder.xor_kernel(W_col.unsqueeze(2), X_col.unsqueeze(0))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)

        return grad_W_col, grad_X_col


class adder2d(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output
