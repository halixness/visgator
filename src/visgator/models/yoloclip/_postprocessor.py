##
##
##

from visgator.models import PostProcessor as _PostProcessor
from visgator.utils.bbox import BBoxes


class PostProcessor(_PostProcessor[BBoxes]):
    def forward(self, output: BBoxes) -> BBoxes:
        return output
