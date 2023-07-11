##
##
##

import torch
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor, nn
from transformers.models.clip.modeling_clip import (
    CLIPTextModelWithProjection,
)
from transformers.models.clip.processing_clip import CLIPProcessor

from ._misc import Models


class TextEncoder(nn.Module):
    """A wrapper around the CLIP [1]_ text encoder.

    .. note::
        To make the output dimension of the text encoder match the output dimension
        of the vision encoder, if the specified `output_dim` is different from
        the dimension of the text encoder's projection layer, the projection layer
        is replaced with a new linear layer that has the specified output dimension.
        The weights of the linear layer are initialized by applying PCA to the weights
        of the original projection layer.

    References
    ----------
    .. [1] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal,
        S., Sastry, G., Askell, A., Mishkin, P., Clark, J. and Krueger, G., 2021, July.
        Learning transferable visual models from natural language supervision.
        In International conference on machine learning (pp. 8748-8763). PMLR."""

    def __init__(self, model: Models, output_dim: int) -> None:
        super().__init__()

        self._dummy = nn.Parameter(torch.empty(0))

        self.processor = CLIPProcessor.from_pretrained(model.weights())
        self.transformer = CLIPTextModelWithProjection.from_pretrained(model.weights())

        projection = self.transformer.text_projection
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

    def forward(self, text: list[str]) -> Float[Tensor, "N D"]:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self._dummy.device)
        attention_mask = inputs["attention_mask"].to(self._dummy.device)

        x = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        text_embeds = x.text_embeds  # (N, D)
        out: Tensor = self.projection(text_embeds)

        return out

    def __call__(self, text: list[str]) -> Float[Tensor, "N D"]:
        """Encodes each text in the batch into a vector.

        Parameters
        ----------
        text : list[str]
            A list of texts to encode.

        Returns
        -------
        Float[Tensor, "N D"]
            Tensor of shape (N, D) where N is the number of texts in the batch and D
            is the output dimension of the text encoder.
        """

        return super().__call__(text)  # type: ignore
