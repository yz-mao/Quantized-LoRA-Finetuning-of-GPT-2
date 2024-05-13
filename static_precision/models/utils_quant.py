# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                if not input.is_contiguous():
                    input = input.contiguous()

                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizeLayer():
    def __init__(self, symmetric=True, w_bits=32, a_bits=32, act_layerwise=False, weight_layerwise=False):

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        self.symmetric = symmetric

        if self.a_bits < 32 and self.a_bits > 2:
            if symmetric:
                self.act_quantizer = SymQuantizer
            else:
                self.act_quantizer = AsymQuantizer

    def quantize_weights(self, real_weights):

        if self.w_bits >= 32:
            return real_weights

        elif self.w_bits >= 3:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            return SymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            )

        elif self.w_bits == 1:
            if self.weight_layerwise:
                scaling_factor = torch.mean(abs(real_weights)).detach()
            else:
                scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True).detach()

            quan_weights_no_grad = scaling_factor * (torch.sign(real_weights / scaling_factor))

        else:
            num_bits = 2 ** (self.w_bits - 1)
            clip_val = 1 - 1e-2
            if self.weight_layerwise:
                scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
            else:
                scaling_factor = (
                        2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                )

            quan_weights_no_grad = (
                    scaling_factor
                    * (
                            torch.round(
                                torch.clamp(real_weights / scaling_factor, -clip_val, clip_val)
                                * num_bits - 0.5
                            )
                            + 0.5
                    )
                    / num_bits
            )

        # 计算权重时需要分离量化和真实权重
        return quan_weights_no_grad.detach() - real_weights.detach() + real_weights

    def quantize_inputs(self, x):

        if self.a_bits < 32 and self.a_bits > 2:
            act_clip_val = torch.tensor([-2.0, 2.0])
            return self.act_quantizer.apply(
                x, act_clip_val, self.a_bits, self.act_layerwise
            )
        return x


class QuantizeConv1D(QuantizeLayer, nn.Module):

    def __init__(self, nf, nx, symmetric=True, w_bits=32, a_bits=32, act_layerwise=False, weight_layerwise=False):

        nn.Module.__init__(self)

        # 然后初始化 QuantizeLayer，传递额外的参数
        QuantizeLayer.__init__(self, symmetric, w_bits, a_bits, act_layerwise, weight_layerwise)

        self.nf = nf
        self.nx = nx

        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        weight = self.quantize_weights(self.weight)
        x = self.quantize_inputs(x)

        # 类似 Conv1D 的操作
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)

        return x


class QuantizeLinear(QuantizeLayer, nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        symmetric=True,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
        bias=True
    ):

        nn.Linear.__init__(self, in_features, out_features, bias=bias)

        # 然后初始化 QuantizeLayer，传递额外的参数
        QuantizeLayer.__init__(self, symmetric, w_bits, a_bits, act_layerwise, weight_layerwise)


    def forward(self, input_):
        weight = self.quantize_weights(self.weight)
        input_ = self.quantize_inputs(input_)

        out = nn.functional.linear(input_, weight)

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
