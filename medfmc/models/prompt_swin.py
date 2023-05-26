import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from copy import deepcopy
from mmcls.models import BACKBONES
from mmcls.models.backbones.swin_transformer import (SwinBlock,
                                                     SwinBlockSequence,
                                                     SwinTransformer)
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.utils.attention import ShiftWindowMSA, WindowMSA
from mmcv.cnn.bricks.transformer import (AdaptivePadding, PatchEmbed,
                                         PatchMerging)
from mmcv.runner.base_module import BaseModule, ModuleList
from typing import List, Sequence


class PromptedPatchMerging(PatchMerging):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map ((used in Swin Transformer)).
    Our implementation uses `nn.Unfold` to
    merge patches, which is about 25% faster than the original
    implementation. However, we need to modify pretrained
    models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=None,
        padding='corner',
        dilation=1,
        bias=False,
        norm_cfg=dict(type='LN'),
        init_cfg=None,
        prompt_length=1,
        prompt_pos='prepend',
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos

    def upsample_prompt(self, prompt_emb):
        prompt_emb = torch.cat(
            (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size

        if self.prompt_pos == 'prepend':
            # change input size
            prompt_emb = x[:, :self.prompt_length, :]
            x = x[:, self.prompt_length:, :]
            L = L - self.prompt_length
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        # add the prompt back:
        if self.prompt_pos == 'prepend':
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x, output_size


class PromptedWindowMSA(WindowMSA):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 prompt_length,
                 prompt_pos,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):
        super().__init__(
            embed_dims=embed_dims,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            init_cfg=init_cfg)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])
        if self.prompt_pos == 'prepend':
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape

            relative_position_bias = torch.cat(
                (torch.zeros(_C, self.prompt_length, _W,
                             device=attn.device), relative_position_bias),
                dim=1)
            relative_position_bias = torch.cat((torch.zeros(
                _C,
                _H + self.prompt_length,
                self.prompt_length,
                device=attn.device), relative_position_bias),
                                               dim=-1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            if self.prompt_pos == 'prepend':
                # expand relative_position_bias
                mask = torch.cat((torch.zeros(
                    nW, self.prompt_length, _W, device=attn.device), mask),
                                 dim=1)
                mask = torch.cat((torch.zeros(
                    nW,
                    _H + self.prompt_length,
                    self.prompt_length,
                    device=attn.device), mask),
                                 dim=-1)

            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class PromptedShiftWindowMSA(ShiftWindowMSA):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        version (str, optional): Version of implementation of Swin
            Transformers. Defaults to `v1`.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 pad_small_map=False,
                 input_resolution=None,
                 auto_pad=None,
                 window_msa=WindowMSA,
                 msa_cfg=dict(),
                 init_cfg=None,
                 prompt_length=1,
                 prompt_pos='prepend'):
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            dropout_layer=dropout_layer,
            pad_small_map=pad_small_map,
            input_resolution=input_resolution,
            auto_pad=auto_pad,
            window_msa=window_msa,
            msa_cfg=msa_cfg,
            init_cfg=init_cfg)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos
        if self.prompt_pos == 'prepend':
            self.w_msa = PromptedWindowMSA(
                prompt_length,
                prompt_pos,
                embed_dims=embed_dims,
                window_size=to_2tuple(window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                **msa_cfg,
            )

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape

        if self.prompt_pos == 'prepend':
            # change input size
            prompt_emb = query[:, :self.prompt_length, :]
            query = query[:, self.prompt_length:, :]
            L = L - self.prompt_length

        assert L == H * W, f"The query length {L} doesn't match the input "\
            f'shape ({H}, {W}).'
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        if min(H, W) == window_size:
            # If not pad small feature map, avoid shifting when the window size
            # is equal to the size of feature map. It's to align with the
            # behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0
        elif min(H, W) < window_size:
            # In the original implementation, the window size will be shrunk
            # to the size of feature map. The behavior is different with
            # swin-transformer for downstream tasks. To support dynamic input
            # shape, we don't allow this feature.
            assert self.pad_small_map, \
                f'The input shape ({H}, {W}) is smaller than the window ' \
                f'size ({window_size}). Please set `pad_small_map=True`, or ' \
                'decrease the `window_size`.'

        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if shift_size > 0:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self.get_attn_mask((H_pad, W_pad),
                                       window_size=window_size,
                                       shift_size=shift_size,
                                       device=query.device)

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query, window_size)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size**2, C)

        # add back the prompt for attn for parralel-based prompts
        # nW*B, prompt_length + window_size*window_size, C
        num_windows = int(query_windows.shape[0] / B)
        if self.prompt_pos == 'prepend':
            # expand prompts_embs
            # B, prompt_length, C --> nW*B, prompt_length, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.prompt_length, C))
            query_windows = torch.cat((prompt_emb, query_windows), dim=1)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # seperate prompt embs --> nW*B, prompt_length, C
        if self.prompt_pos == 'prepend':
            # change input size
            prompt_emb = attn_windows[:, :self.prompt_length, :]
            attn_windows = attn_windows[:, self.prompt_length:, :]
            # change prompt_embs's shape:
            # nW*B, prompt_length, C - B, prompt_length, C
            prompt_emb = prompt_emb.view(-1, B, self.prompt_length, C)
            prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad,
                                        window_size)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if H != H_pad or W != W_pad:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # add the prompt back:
        if self.prompt_pos == 'prepend':
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.drop(x)

        return x


class PromptedSwinBlock(SwinBlock):
    """Swin Transformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        shift (bool): Shift the attention window or not. Defaults to False.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 shift=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 prompt_length=1,
                 prompt_pos='prepend'):
        super(PromptedSwinBlock, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift=shift,
            ffn_ratio=ffn_ratio,
            drop_path=drop_path,
            pad_small_map=pad_small_map,
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            init_cfg=init_cfg,
        )
        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            'prompt_length': prompt_length,
            'prompt_pos': prompt_pos,
            **attn_cfgs
        }
        self.attn = PromptedShiftWindowMSA(**_attn_cfgs)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class PromptedSwinBlockSequence(SwinBlockSequence):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        downsample_cfg (dict): The extra config of the patch merging layer.
            Defaults to empty dict.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=7,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 init_cfg=None,
                 prompt_length=1,
                 prompt_pos='prepend'):
        super().__init__(
            embed_dims=embed_dims,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            downsample=downsample,
            downsample_cfg=downsample_cfg,
            drop_paths=drop_paths,
            block_cfgs=block_cfgs,
            with_cp=with_cp,
            pad_small_map=pad_small_map,
            init_cfg=init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'prompt_length': prompt_length,
                'prompt_pos': prompt_pos,
                **block_cfgs[i]
            }
            block = PromptedSwinBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': 2 * embed_dims,
                'norm_cfg': dict(type='LN'),
                'prompt_length': prompt_length,
                'prompt_pos': prompt_pos,
                **downsample_cfg
            }
            self.downsample = PromptedPatchMerging(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        if self.prompt_length is not None:
            for block in self.blocks:
                x = block(x, in_shape)

        if self.downsample is not None and do_downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.embed_dims


@BACKBONES.register_module()
class PromptedSwinTransformer(SwinTransformer):

    def __init__(
        self,
        arch='base',
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=7,
        drop_path_rate=0.1,
        with_cp=False,
        pad_small_map=False,
        stage_cfgs=dict(),
        patch_cfg=dict(),
        prompt_length=1,
        prompt_layers=None,
        prompt_pos='prepend',
        prompt_init='normal',
    ):
        super().__init__(arch=arch)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'prompt_length': prompt_length,
                'prompt_pos': prompt_pos,
                **stage_cfg
            }
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=img_size,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = PatchEmbed(**_patch_cfg)

            stage = PromptedSwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)
            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)
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

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        # Add prompt
        if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
            with torch.no_grad():
                self.prompt.data += x.mean([0, 1]).detach().clone()
            self.prompt_initialized = True
        prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        # prompt: [layer, batch, length, dim]
        if self.prompt_pos == 'prepend':
            # x = torch.cat([x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
            #               dim=1)
            x = torch.cat([prompt[0, :, :, :], x], dim=1)
            # vpt_swin: (batch_size, n_prompt + n_patches, hidden_dim)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                # out = out.view(-1, *hw_shape,
                #                self.num_features[i]).permute(0, 3, 1,
                #                                              2).contiguous()

                out = self.avgpool(out.transpose(1, 2))  # B C 1
                out = torch.flatten(out, 1)
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and \
                self.__class__ is PromptedSwinTransformer:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None or version < 3) and \
                self.__class__ is PromptedSwinTransformer:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)
