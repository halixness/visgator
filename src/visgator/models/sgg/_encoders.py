##
##
##

from typing import Callable, Optional, Union

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from jaxtyping import Bool, Float, Integer
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer
from open_clip.transformer import (
    ResidualAttentionBlock as OpenClipResidualAttentionBlock,
)
from open_clip.transformer import Transformer as OpenClipTransformer
from open_clip.transformer import VisionTransformer as OpenClipVisionTransformer
from torch import LongTensor, Tensor, nn

from visgator.utils.batch import Batch
from visgator.utils.torch import Nested4DTensor

from ._config import EncodersConfig
from ._misc import CaptionEmbeddings

# ------------------------------------------------------------------------------
# Vision Encoder
# ------------------------------------------------------------------------------


class VisionEncoder(nn.Module):
    def __init__(
        self,
        encoder: OpenClipVisionTransformer,
        output_dim: int,
        mean: Optional[tuple[float, float, float]] = None,
        std: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()

        if encoder.input_patchnorm:
            raise NotImplementedError

        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1))
        self._mean: Tensor  # just for type hinting
        self._std: Tensor  # just for type hinting

        self._patch_conv = encoder.conv1

        self._class_emb = encoder.class_embedding  # D
        self._positional_emb = encoder.positional_embedding  # (G*G + 1) D

        self._patch_dropout = encoder.patch_dropout

        self._pre_ln = encoder.ln_pre

        self._transformer = _VisionTransformer(encoder.transformer)

        self._attn_pool = encoder.attn_pool
        if self._attn_pool is not None:
            raise NotImplementedError

        self._post_ln = encoder.ln_post

        self._freeze()

        clip_output = encoder.proj.shape[0]
        if output_dim != encoder.proj.shape[1]:
            # create a new trainable projection matrix
            width = encoder.conv1.weight.shape[0]
            scale = width**-0.5
            self._proj = nn.Parameter(scale * torch.randn(clip_output, output_dim))
        else:
            self._proj = encoder.proj
            self._proj.requires_grad_(False)

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images: Nested4DTensor) -> Nested4DTensor:
        img_tensor = (images.tensor - self._mean) / self._std
        img_tensor.masked_fill_(images.mask.unsqueeze(1).expand(-1, 3, -1, -1), 0.0)
        images = Nested4DTensor(img_tensor, images.sizes, images.mask)

        x = self._patch_conv(images.tensor)
        H, W = x.shape[-2:]  # H W
        new_masks = F.interpolate(images.mask[None].float(), size=x.shape[-2:])[0]
        new_masks = new_masks.bool()  # N H W

        patch_embeddings = self._positional_emb[1:]
        num_patches = patch_embeddings.shape[0]
        side = int(num_patches**0.5)
        patch_embeddings = patch_embeddings.view(1, side, side, -1).permute(0, 3, 1, 2)

        new_sizes = []
        pos_embs = torch.zeros_like(x)  # N D H W

        kernel_height, kernel_wodth = self._patch_conv.kernel_size
        padding_height, padding_width = self._patch_conv.padding
        stride_height, stride_width = self._patch_conv.stride
        for i in range(len(x)):
            shape = images.sizes[i]
            h = ((shape[0] + 2 * padding_height - kernel_height) // stride_height) + 1
            w = ((shape[1] + 2 * padding_width - kernel_wodth) // stride_width) + 1

            new_sizes.append((h, w))

            emb = F.interpolate(
                patch_embeddings,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            emb = emb[0]  # D H W

            pos_embs[i, :, :h, :w] = emb

        pos_embs = pos_embs.flatten(2).transpose(1, 2)  # N (HW) D
        x = x.flatten(2).transpose(1, 2)  # N (HW) D
        masks = new_masks.flatten(1)  # N (HW)

        cls_pos_emb = self._positional_emb[0][None, None, :].expand(len(x), -1, -1)
        pos_embs = torch.cat([cls_pos_emb, pos_embs], dim=1)  # N (1+HW) D

        cls_token = self._class_emb[None, None, :].expand(len(x), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # N (1+HW) D
        x = x + pos_embs

        masks = torch.cat([masks.new_zeros(len(x), 1), masks], dim=1)  # N (1+HW)

        x = self._patch_dropout(x)
        x = self._pre_ln(x)

        x = x.transpose(0, 1)  # N L D -> L N D
        x = self._transformer(x, masks)
        x = x.transpose(0, 1)  # L N D -> N L D

        if self._attn_pool is not None:
            x = self._attn_pool(x, masks)

        x = self._post_ln(x)
        x = x @ self._proj

        x = x[:, 1:]  # remove cls token
        x = x.view(len(x), H, W, -1)
        x = x.permute(0, 3, 1, 2)  # N D H W

        return Nested4DTensor(x, new_sizes, new_masks)

    def __call__(self, images: Nested4DTensor) -> Nested4DTensor:
        return super().__call__(images)  # type: ignore


class _VisionTransformer(nn.Module):
    def __init__(self, transformer: OpenClipTransformer) -> None:
        super().__init__()

        self._blocks = nn.ModuleList(
            [_ResidualAttentionBlock(block) for block in transformer.resblocks]
        )

    def forward(
        self,
        x: Float[Tensor, "L N D"],
        padding_mask: Bool[Tensor, "N L"],
    ) -> Float[Tensor, "L N D"]:
        for block in self._blocks:
            x = block(x, padding_mask)

        return x


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, block: OpenClipResidualAttentionBlock) -> None:
        super().__init__()

        self._ln_1 = block.ln_1
        self._ls_1 = block.ls_1

        self._attn = block.attn

        self._ln_2 = block.ln_2
        self._ls_2 = block.ls_2

        self._mlp = block.mlp

    def forward(
        self,
        x: Float[Tensor, "L N D"],
        padding_mask: Bool[Tensor, "N L"],
    ) -> Float[Tensor, "L N D"]:
        tmp = self._ln_1(x)
        tmp, _ = self._attn(
            tmp,
            tmp,
            tmp,
            need_weights=False,
            key_padding_mask=padding_mask,
        )

        x = x + self._ls_1(tmp)
        x = x + self._ls_2(self._mlp(self._ln_2(x)))

        return x


# ------------------------------------------------------------------------------
# Text Encoder
# ------------------------------------------------------------------------------

Tokenizer = Union[HFTokenizer, Callable[[Union[str, list[str]]], LongTensor]]


class TextEncoder(nn.Module):
    def __init__(self, model: CLIP, tokenizer: Tokenizer, output_dim: int) -> None:
        super().__init__()

        self._dummy = nn.Parameter(torch.empty(0))

        self._tokenizer = tokenizer

        self._transformer = model.transformer
        self._context_length = model.context_length
        self._token_emb = model.token_embedding
        self._positional_emb = model.positional_embedding
        self._final_ln = model.ln_final
        self.register_buffer("_attn_mask", model.attn_mask)
        self._attn_mask: Tensor  # just for type checking

        self._freeze()

        if output_dim == model.text_projection.shape[1]:
            self._text_projection = model.text_projection
            self._text_projection.requires_grad_(False)
        else:
            self._text_projection = nn.Parameter(
                torch.randn(model.text_projection.shape[0], output_dim)
            )

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _flatten(self, batch: Batch) -> Integer[Tensor, "N L"]:
        texts = []
        for sample in batch:
            caption = sample.caption
            assert caption.graph is not None

            entities = caption.graph.entities
            relations = caption.graph.relations

            texts.append(caption.sentence)
            texts.extend([entity.span for entity in entities])
            texts.extend(
                [
                    f"{entities[rel.subject].span} {rel.predicate} "
                    f"{entities[rel.object].span}"
                    for rel in relations
                ]
            )

        return self._tokenizer(texts).to(self._dummy.device)

    def _unflatten(self, batch: Batch, tokens: Tensor) -> list[CaptionEmbeddings]:
        captions = []

        idx = 0
        for sample in batch:
            graph = sample.caption.graph
            assert graph is not None

            sentence = tokens[idx]
            idx += 1

            entities = tokens[idx : idx + len(graph.entities)]
            idx += len(graph.entities)

            relations = tokens[idx : idx + len(graph.relations)]
            idx += len(graph.relations)

            captions.append(
                CaptionEmbeddings(
                    sentence=sentence,
                    entities=entities,
                    relations=relations,
                )
            )

        return captions

    def forward(self, batch: Batch) -> list[CaptionEmbeddings]:
        text = self._flatten(batch)

        cast_dtype = self._transformer.get_cast_dtype()
        x = self._token_emb(text).to(cast_dtype)
        x = x + self._positional_emb.to(cast_dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self._transformer(x, attn_mask=self._attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self._final_ln(x)
        x = x[torch.arange(x.shape[0]), text.argmax(-1)] @ self._text_projection

        return self._unflatten(batch, x)

    def __call__(self, batch: Batch) -> list[CaptionEmbeddings]:
        return super().__call__(batch)  # type: ignore


# ------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------


def build_encoders(config: EncodersConfig) -> tuple[VisionEncoder, TextEncoder]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model,
        pretrained=config.pretrained,
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    if type(preprocess) is not T.Compose:
        raise NotImplementedError

    mean: Optional[tuple[float, float, float]] = None
    std: Optional[tuple[float, float, float]] = None
    for transform in preprocess.transforms:
        if type(transform) is T.Normalize:
            mean = transform.mean
            std = transform.std
            break

    vision = VisionEncoder(model.visual, config.hidden_dim, mean, std)
    text = TextEncoder(model, tokenizer, config.hidden_dim)

    return vision, text
