import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones import VisionTransformer
from mmpretrain.models.utils import resize_pos_embed
from mmpretrain.models.utils import build_norm_layer

@MODELS.register_module()
class PromptedViT(VisionTransformer):
    '''
    
    prompt_length (int):
    deep_prompt (bool):
    prompt_init (str):
    '''

    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap', 'avg_all', 'avg_prompt', 'avg_prompt_clstoken'}
    # 'avg_all' : avg of 'prompt' & 'cls_token' & 'featmap'
    # 'avg_prompt' avg of 'prompt'
    # 'avg_prompt_clstoken' avg of 'cls_token' and 'prompt'
    def __init__(self,
                 prompt_length = 1,
                 deep_prompt = True,
                 out_type='avg_all',
                 prompt_init: str = 'normal',
                 norm_cfg=dict(type='LN'),
                 *args,
                 **kwargs):
        super().__init__(*args, out_type=out_type,  norm_cfg=norm_cfg, **kwargs)

        self.prompt_layers = len(self.layers) if deep_prompt else 1
        prompt = torch.empty(
            self.prompt_layers, prompt_length, self.embed_dims)
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
        self.deep_prompt = deep_prompt

        if self.out_type in {'avg_featmap', 'avg_all', 'avg_prompt', 'avg_prompt_clstoken'}:
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)
        
        # freeze stages 
        self.frozen_stages = len(self.layers)
        self._freeze_stages()

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        # reshape to [layers, batch, tokens, embed_dims]
        prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        x = torch.cat(
                [x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
                dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.deep_prompt and i != len(self.layers) - 1:
                x = torch.cat(
                        [x[:, :1, :], prompt[i, :, :, :], x[:, self.prompt_length + 1:, :]],
                        dim=1)

            # final_norm should be False here
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(x[:, self.prompt_length:].mean(dim=1))     
        if self.out_type == 'avg_all':
            return self.ln2(x.mean(dim=1))  
        if self.out_type == 'avg_prompt':
            return self.ln2(x[:, 1:self.prompt_length+1].mean(dim=1))  
        if self.out_type == 'avg_prompt_clstoken':
            return self.ln2(x[:, :self.prompt_length+1].mean(dim=1))  
         
