import torch
import torch.nn as nn
from mmcls.models import BACKBONES
from mmcls.models.backbones import VisionTransformer
from mmcls.models.utils import resize_pos_embed
from typing import List


@BACKBONES.register_module()
class PromptedVisionTransformer(VisionTransformer):

    def __init__(self,
                 prompt_length: int = 1,
                 prompt_layers: List[int] = None,
                 prompt_pos: str = 'prepend',
                 prompt_init: str = 'normal',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
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
        """Following mmcls implementation."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

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

        # Add prompt
        if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
            with torch.no_grad():
                self.prompt.data += x.mean([0, 1]).detach().clone()
            self.prompt_initialized = True
        prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        # prompt: [layer, batch, length, dim]
        if self.prompt_pos == 'prepend':
            x = torch.cat([x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
                          dim=1)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            if i in self.prompt_layers:
                if self.prompt_pos == 'prepend':
                    x = torch.cat([
                        x[:, :1, :], prompt[i, :, :, :],
                        x[:, 1 + self.prompt_length:, :]
                    ],
                                  dim=1)
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                outs.append(x[:, 0])

        return tuple(outs)
