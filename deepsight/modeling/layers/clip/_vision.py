##
##
##

import torch
import torch.nn.functional as F
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor, nn
from transformers.models.clip.modeling_clip import (
    CLIPVisionEmbeddings,
    CLIPVisionModelWithProjection,
)

from deepsight.utils.torch import Batched2DTensors, Batched3DTensors

from ._misc import Models


class VisionEncoder(nn.Module):
    """A modified version of the CLIP [1]_ vision encoder.

    There are two main differences:
    - While the CLIP vision encoder returns a single vector representation for each
    image by pooling the patch embeddings, this encoder returns a 2D feature map for
    each image. Similarly to OwlViT [2]_, the 2D feature map is obtained by multiplying
    the patches with the class token and applying a layer norm. Each patch is then
    projected to the output dimension using a linear layer.
    - While the CLIP vision encoder requires all images to be rescaled to the same
    size (224x224 or 336x336), this encoder does not require images to be rescaled to
    the same fixed size. Instead, the positional embeddings are interpolated to the size
    of each image. This should improve the performance of the encoder on large images
    with fine-grained details.

    .. note::
        The CLIP vision encoder has an output dimension of 512 that is double what is
        used by most object detection models. Thus, if the specified `output_dim` is
        not 512, the projection layer is replaced with a linear layer that has the
        specified output dimension. The weights of the linear layer are initialized by
        applying PCA to the weights of the original projection layer.

    References
    ----------
    .. [1] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal,
        S., Sastry, G., Askell, A., Mishkin, P., Clark, J. and Krueger, G., 2021, July.
        Learning transferable visual models from natural language supervision.
        In International conference on machine learning (pp. 8748-8763). PMLR.
    .. [2] Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D.,
        Dosovitskiy, A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z. and Wang, X.,
        2022, October. Simple open-vocabulary object detection. In European Conference
        on Computer Vision (pp. 728-755). Cham: Springer Nature Switzerland.
    """

    def __init__(self, model: Models, output_dim: int) -> None:
        super().__init__()

        clip = CLIPVisionModelWithProjection.from_pretrained(model.weights())

        vision = clip.vision_model
        self.embeddings = VisionEmbeddings(vision.embeddings)
        self.pre_layernorm = vision.pre_layrnorm
        self.encoder = vision.encoder
        self.post_layernorm = vision.post_layernorm

        self.last_layernorm = nn.LayerNorm(clip.config.hidden_size)

        projection = clip.visual_projection
        if projection.out_features != output_dim:
            weights = projection.weight.transpose(0, 1).detach().numpy()
            weights = PCA(output_dim).fit_transform(weights)
            self.projection = nn.Linear(
                in_features=projection.in_features,
                out_features=output_dim,
                bias=False,
            )

            with torch.no_grad():
                self.projection.weight = nn.Parameter(
                    torch.from_numpy(weights).transpose(0, 1)
                )

        else:
            self.projection = projection

    def _create_attention_mask(self, x: Batched2DTensors) -> Float[Tensor, "B 1 L L"]:
        """Creates an attention mask to mask out the padding tokens.

        Parameters
        ----------
        x : Batched2DTensors
            The input flattened image tensors.

        Returns
        -------
        Float[Tensor, "B 1 L L"]
            The attention mask.
        """

        mask = x.mask[:, None, None, :].expand(-1, 1, x.shape[1], -1)

        dtype = x.tensor.dtype
        attn_mask = torch.zeros_like(mask, dtype=dtype)
        attn_mask.masked_fill_(mask, -torch.inf)

        return attn_mask

    def forward(self, images: Batched3DTensors) -> Batched3DTensors:
        x, new_sizes = self.embeddings(images)  # (B, 1+HW, C)

        attn_mask = self._create_attention_mask(x)

        hidden: Tensor = self.pre_layernorm(x.tensor)
        tmp = self.encoder(
            inputs_embeds=hidden,
            output_attentions=False,
            output_hidden_states=False,
            attention_mask=attn_mask,
            return_dict=True,
        )

        hidden = tmp.last_hidden_state  # (B, 1+HW, C)
        hidden = self.post_layernorm(hidden)

        class_token = hidden[:, :1]  # (B, 1, C)
        image_embeds = hidden[:, 1:]  # (B, HW, C)

        out: Tensor = class_token * image_embeds
        out = self.last_layernorm(out)

        out = self.projection(out)  # (B, HW, D)

        H = max(size[0] for size in new_sizes)
        W = max(size[1] for size in new_sizes)

        out = out.view(out.shape[0], H, W, -1).permute(0, 3, 1, 2)  # (B, D, H, W)

        return Batched3DTensors(out, sizes=new_sizes)

    def __call__(self, images: Batched3DTensors) -> Batched3DTensors:
        return super().__call__(images)  # type: ignore


class VisionEmbeddings(nn.Module):
    """A wrapper around the CLIP vision embeddings.

    This wrapper allows the CLIP vision encoder to work with batches of images of
    different sizes. To avoid discarding the learned positional embeddings, the
    positional embeddings are interpolated to the size of each image.
    """

    def __init__(self, embeddings: CLIPVisionEmbeddings) -> None:
        super().__init__()

        self.patch_embedding = embeddings.patch_embedding
        self.class_embedding = embeddings.class_embedding

        h, w = (int(embeddings.num_patches**0.5),) * 2
        patch_pos_embedding = embeddings.position_embedding.weight.data[1:]
        patch_pos_embedding = patch_pos_embedding.reshape(h, w, -1).permute(2, 0, 1)
        class_pos_embedding = embeddings.position_embedding.weight.data[0]

        self.patch_pos_embedding = nn.Parameter(patch_pos_embedding)
        self.class_pos_embedding = nn.Parameter(class_pos_embedding)

    def _compute_new_size(self, old_size: tuple[int, int]) -> tuple[int, int]:
        """Computes the new size of the image after patch embedding.

        Parameters
        ----------
        old_size : tuple[int, int]
            The size of the image before patch embedding.

        Returns
        -------
        tuple[int, int]
            The size of the image after patch embedding.
        """

        kh, kw = self.patch_embedding.kernel_size
        sh, sw = self.patch_embedding.stride
        ph, pw = self.patch_embedding.padding

        H, W = old_size
        h = (H + 2 * ph - kh) // sh + 1
        w = (W + 2 * pw - kw) // sw + 1

        return h, w

    def forward(
        self, images: Batched3DTensors
    ) -> tuple[Batched2DTensors, list[tuple[int, int]]]:
        B = len(images)
        x: Tensor = self.patch_embedding(images.tensor)

        new_sizes = []
        patch_pos_emb = torch.zeros_like(x)  # (B, C, H, W)

        for idx in range(len(x)):
            h, w = self._compute_new_size(images.sizes[idx])
            new_sizes.append((h, w))

            emb = F.interpolate(
                self.patch_pos_embedding[None],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]

            patch_pos_emb[idx, :, :h, :w] = emb

        patch_pos_emb = patch_pos_emb.flatten(2).transpose(1, 2)  # (B, HW, C)
        class_pos_emb = self.class_pos_embedding.expand(B, 1, -1)  # (B, 1, C)
        pos_emb = torch.cat([class_pos_emb, patch_pos_emb], dim=1)  # (B, 1+HW, C)

        class_token = self.class_embedding.expand(B, 1, -1)  # (B, 1, C)
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = torch.cat([class_token, x], dim=1)  # (B, 1+HW, C)

        x = x + pos_emb

        out = Batched2DTensors(x, sizes=[(1 + h * w) for h, w in new_sizes])

        return out, new_sizes

    def __call__(
        self, images: Batched3DTensors
    ) -> tuple[Batched2DTensors, list[tuple[int, int]]]:
        return super().__call__(images)  # type: ignore
