from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch import nn

    from einops import rearrange
    from einops.layers.torch import Rearrange

    from benchmark_utils.debfly.debfly_interface import DeBflyLinear

    from benchmark_utils.debfly.generalized_fly_utils import (
        compute_monarch_r_shape,
        get_i_th_monotone_chain_min_params,
        DebflyGen,
        get_low_rank_chain,
    )

    import numpy as np


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def calculate_shape_debfly_feed_forward(
    dim,
    hidden_dim,
    chain_type,
    monarch_blocks=None,
    num_debfly_factors=None,
    rank=None,
    chain_idx=None,
    format="abcdpq",
):
    result = []
    first_generator = DebflyGen(hidden_dim, dim, rank)  # outsize, insize, rank
    second_generator = DebflyGen(
        dim, hidden_dim, rank
    )  # outsize, insize, rank

    if chain_type == "monotone_min_param":
        assert (
            num_debfly_factors is not None
            and rank is not None
            and chain_idx is not None
        )
        if chain_idx == 0:
            _, chain = first_generator.smallest_monotone_debfly_chain(
                num_debfly_factors, format=format
            )
        else:
            chain = get_i_th_monotone_chain_min_params(
                first_generator, num_debfly_factors, rank, chain_idx
            )
        result.append(chain[::-1])
        if chain_idx == 0:
            _, chain = second_generator.smallest_monotone_debfly_chain(
                num_debfly_factors, format=format
            )
        else:
            chain = get_i_th_monotone_chain_min_params(
                second_generator, num_debfly_factors, rank, chain_idx
            )
        result.append(chain[::-1])
    elif chain_type == "monarch":
        assert monarch_blocks is not None
        result.append(
            compute_monarch_r_shape(
                dim, hidden_dim, monarch_blocks, format=format
            )
        )
        result.append(
            compute_monarch_r_shape(
                hidden_dim, dim, monarch_blocks, format=format
            )
        )
    elif chain_type == "random":
        chain = first_generator.random_debfly_chain(
            n_factors=num_debfly_factors, format=format
        )
        result.append(chain[::-1])
        chain = second_generator.random_debfly_chain(
            n_factors=num_debfly_factors, format=format
        )
        result.append(chain[::-1])
    elif chain_type == "low-rank":
        assert rank is not None
        result.append(
            get_low_rank_chain(dim, hidden_dim, rank, format=format)[::-1]
        )
        result.append(
            get_low_rank_chain(hidden_dim, dim, rank, format=format)[::-1]
        )
    else:
        raise NotImplementedError
    return result


class DebflyFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, deblfy_layer, version, **debfly_cfg):
        super().__init__()
        assert len(deblfy_layer) == 2
        R_shapes_list = calculate_shape_debfly_feed_forward(
            dim, hidden_dim, **debfly_cfg
        )
        if deblfy_layer[0]:
            linear1 = DeBflyLinear(
                dim, hidden_dim, R_shapes_list[0], version=version
            )  # insize, outsize
        else:
            linear1 = nn.Linear(dim, hidden_dim)
        if deblfy_layer[1]:
            linear2 = DeBflyLinear(
                hidden_dim, dim, R_shapes_list[1], version=version
            )  # insize, outsize
        else:
            linear2 = nn.Linear(hidden_dim, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), linear1, nn.GELU(), linear2
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        debfly_layer_lists=None,
        version="densification",
        **debfly_cfg
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        if debfly_layer_lists is not None:
            assert len(debfly_layer_lists) == depth
            for i in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, heads=heads, dim_head=dim_head),
                            DebflyFeedForward(
                                dim,
                                mlp_dim,
                                debfly_layer_lists[i],
                                version,
                                **debfly_cfg
                            ),
                        ]
                    )
                )
        else:
            for i in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, heads=heads, dim_head=dim_head),
                            FeedForward(dim, mlp_dim),
                        ]
                    )
                )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        debfly_layer_lists=None,
        version="densification",
        **debfly_cfg
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # num_patches = (image_height // patch_height) * (
        #     image_width // patch_width
        # )
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            debfly_layer_lists,
            version,
            **debfly_cfg
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # *_, h, w, dtype = *img.shape, img.dtype
        *_, _, _, _ = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


def simple_vit_s16_in1k():
    dim = 384
    heads = 6
    return SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=dim,
        depth=12,
        heads=heads,
        mlp_dim=4 * dim,
        dim_head=dim // heads,
    )


def simple_vit_b16_in1k():
    dim = 768
    heads = 12
    return SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=dim,
        depth=12,
        heads=heads,
        mlp_dim=4 * dim,
        dim_head=dim // heads,
    )


def simple_vit_s16_in1k_butterfly(num_debfly_layer, version, **debfly_cfg):
    """
    :param num_debfly_layer: int
    :param debfly_cfg: argument for calculate_shape_debfly_feed_forward
    :param version: version for DeBfly forward
    :return: SimpleViT with butterfly structure
    """
    dim = 384
    heads = 6
    depth = 12

    # Constructing debfly_conv_list
    debfly_layer_lists = np.zeros(depth * 2, dtype=np.int8)
    assert 0 <= num_debfly_layer <= len(debfly_layer_lists)
    debfly_layer_lists[-num_debfly_layer:] = np.ones_like(
        debfly_layer_lists[-num_debfly_layer:]
    )
    debfly_layer_lists = debfly_layer_lists.reshape(depth, 2)
    debfly_layer_lists = debfly_layer_lists.tolist()

    return SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=4 * dim,
        dim_head=dim // heads,
        debfly_layer_lists=debfly_layer_lists,
        version=version,
        **debfly_cfg
    )


def simple_vit_b16_in1k_butterfly(num_debfly_layer, version, **debfly_cfg):
    """
    :param num_debfly_layer: int
    :param debfly_cfg: argument for calculate_shape_debfly_feed_forward
    :param version: version for DeBfly forward
    :return: SimpleViT with butterfly structure
    """
    depth = 12
    dim = 768  # hidden dim attention head
    heads = 12

    # Constructing debfly_conv_list
    debfly_layer_lists = np.zeros(depth * 2, dtype=np.int8)
    assert 0 <= num_debfly_layer <= len(debfly_layer_lists)
    debfly_layer_lists[-num_debfly_layer:] = np.ones_like(
        debfly_layer_lists[-num_debfly_layer:]
    )
    debfly_layer_lists = debfly_layer_lists.reshape(depth, 2)
    debfly_layer_lists = debfly_layer_lists.tolist()

    return SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=4 * dim,
        dim_head=dim // heads,
        debfly_layer_lists=debfly_layer_lists,
        version=version,
        **debfly_cfg
    )
