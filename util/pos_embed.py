import numpy as np
import math
import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _infer_extra_tokens(total_tokens, max_try=3):
    """
    Automatically infer the number of extra tokens in the checkpoint, and assume that the number of extra tokens is no more than 3.
    Return the number of extra tokens in the checkpoint model, extra_tokens_ckpt, and the number of patch grids in the checkpoint model.
    """
    # 通常 cls / dist / others 不会超过 3 个
    for extra in range(max_try + 1):
        # 逐个猜测额外token的数量，直到猜中
        num_patch = total_tokens - extra
        g = int(math.isqrt(num_patch))
        if g * g == num_patch:
            return extra, g
    raise RuntimeError("The number of tokens in the checkpoint model does not match the number in the original model, "
                       "so positional embedding interpolation cannot be performed.")


def interpolate_pos_embed(model, checkpoint_model):
    """
    When loading pretrained model weights, adjust the positional embeddings by interpolation based on the current
    model structure, because the positional embeddings in the pretrained model may not match those in the current model.

    The basic idea is that the positional embeddings for the extra tokens do not need to be changed, because they are
    not images and do not contain positional information. The positional embeddings for the image patches need to be
    interpolated, because they contain actual pixel position information.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        new_size = int(num_patches ** 0.5)
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        extra_tokens_ckpt, orig_size = _infer_extra_tokens(pos_embed_checkpoint.shape[-2])

        # Interpolate the positional embeddings of the current model.
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :extra_tokens_ckpt]
            pos_tokens = pos_embed_checkpoint[:, extra_tokens_ckpt:]

            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            if extra_tokens_ckpt > num_extra_tokens:
                extra_tokens = extra_tokens[:, :num_extra_tokens]
            elif extra_tokens_ckpt < num_extra_tokens:
                pad = extra_tokens[:, :1].expand(-1, num_extra_tokens - extra_tokens_ckpt, -1)
                extra_tokens = torch.cat([extra_tokens, pad], dim=1)

            checkpoint_model['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)
