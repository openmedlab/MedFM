from typing import List
from mmpretrain.models.backbones import ViTEVA02 as mmpt_ViTEVA02
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import (RotaryEmbeddingFast, SwiGLUFFN, build_norm_layer,
                                        resize_pos_embed)
from mmpretrain.models.backbones.vit_eva02 import EVA02EndcoderLayer



class PromptViTEVA02(mmpt_ViTEVA02):
    """EVA02 Vision Transformer."""
   
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 prompt_length: int = 1,
                 prompt_layers: List[int] = None,
                 prompt_pos: str = 'prepend',
                 prompt_init: str = 'normal',
                 *args,
                 **kwargs):
        super(PromptViTEVA02, self).__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

        self.prompt_layers = [0] if prompt_layers is None else prompt_layers
        prompt = torch.empty(
            len(self.prompt_layers), prompt_length, self.embed_dims)
        if prompt_init == 'uniform':
            nn.init.uniform_(prompt, -0.08, 0.08)
        elif prompt_init == 'zero':
            nn.init.zeros_(prompt)
        elif prompt_init == 'kaiming':
            nn.init.kaiming_normal_(prompt)
        elif prompt_init == 'token':
            nn.init.zeros_(prompt)
            self.prompt_initialized = False
        else:
            nn.init.normal_(prompt, std=0.02)
        self.prompt = nn.Parameter(prompt, requires_grad=True)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos



    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

