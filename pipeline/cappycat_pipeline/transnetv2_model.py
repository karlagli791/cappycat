"""TransNetV2 network definition (PyTorch), vendored from the reference implementation.

Source: https://github.com/soCzech/TransNetV2 (``inference-pytorch/transnetv2_pytorch.py``)

    MIT License - Copyright (c) 2020 Tomas Soucek
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions: The above
    copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

The weights are the official TF weights converted with the repo's ``convert_weights.py``
(published on the Hugging Face hub as ``Sn4kehead/TransNetV2`` / ``magnusdtd/TransNetV2`` -
the uploads are bit-identical). Changes from upstream: unsupported-option branches and the
training-only stochastic-depth path were dropped, ``NotImplemented`` (not callable) became
``NotImplementedError``, and :class:`TransNetV2SingleFrame` wraps the network for export.

Input: ``uint8`` ``[B, T, 27, 48, 3]`` RGB. Output: single-frame transition *logits* ``[B, T, 1]``
(plus the many-hot head). Apply a sigmoid for probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransNetV2(nn.Module):
    def __init__(self, F=16, L=3, S=2, D=1024, use_many_hot_targets=True, use_frame_similarity=True,
                 use_color_histograms=True, use_mean_pooling=False, dropout_rate=0.5):
        super().__init__()
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)]
            + [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )
        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128
        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None
        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)
        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        one_hot = self.cls_layer1(x)
        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}
        return one_hot


class TransNetV2SingleFrame(nn.Module):
    """Returns only the single-frame logits ``[B, T, 1]`` (what shot detection uses)."""

    def __init__(self, net: TransNetV2):
        super().__init__()
        self.net = net

    def forward(self, frames):
        out = self.net(frames)
        return out[0] if isinstance(out, tuple) else out


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters, shortcut=True, pool_type="avg", stochastic_depth_drop_prob=0.0):
        super().__init__()
        assert pool_type == "max" or pool_type == "avg"
        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
        x = functional.relu(x)
        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                x = (1 - self.stochastic_depth_drop_prob) * x + shortcut  # inference form
            else:
                x = x + shortcut
        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None):
        super().__init__()
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        x = torch.cat([self.Conv3D_1(inputs), self.Conv3D_2(inputs), self.Conv3D_4(inputs), self.Conv3D_8(inputs)], dim=1)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate, separable=True, use_bias=True):
        super().__init__()
        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1), dilation=(dilation_rate, 1, 1),
                              padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3, dilation=(dilation_rate, 1, 1),
                             padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def _lookup(similarities: torch.Tensor, lookup_window: int) -> torch.Tensor:
    """[B, T, T] frame-similarity matrix -> [B, T, lookup_window] band around the diagonal."""
    batch_size, time_window = similarities.shape[0], similarities.shape[1]
    similarities_padded = functional.pad(similarities, [(lookup_window - 1) // 2, (lookup_window - 1) // 2])
    dev = similarities.device
    batch_indices = torch.arange(0, batch_size, device=dev).view([batch_size, 1, 1]).repeat([1, time_window, lookup_window])
    time_indices = torch.arange(0, time_window, device=dev).view([1, time_window, 1]).repeat([batch_size, 1, lookup_window])
    lookup_indices = torch.arange(0, lookup_window, device=dev).view([1, 1, lookup_window]).repeat(
        [batch_size, time_window, 1]) + time_indices
    return similarities_padded[batch_indices, time_indices, lookup_indices]


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128, use_bias=False):
        super().__init__()
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)
        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        return functional.relu(self.fc(_lookup(similarities, self.lookup_window)))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super().__init__()
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.reshape(batch_size * time_window, height * width, 3)
        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)
        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones_like(binned_values, dtype=torch.int32))
        histograms = histograms.view(batch_size, time_window, 512).float()
        return functional.normalize(histograms, p=2, dim=2)

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities = _lookup(similarities, self.lookup_window)
        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


def load_transnetv2(weights_path, device: str = "cpu") -> TransNetV2:
    """Build the network and load a state dict saved with ``torch.save(model.state_dict())``."""
    net = TransNetV2()
    sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    return net.eval().to(device)
