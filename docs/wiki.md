# Visgator
Referring expressions visual grounding wih scene graphs and object detection.

## Architecture
![visgator architecture](src/architecture.png)

For each training sample (image, caption), the idea is to:
1. Identify all the Entity-Relationship-Pairs (ERP) from the caption ([Preprocessing](#preprocessing))
2. Given multiple portions of the image as candidates for each entity, spawn an ERP for all the visual instantiations of each entity ([Instantiation](#instantiation)).
3. Enrich each entity-pair (a sequence of visual tokens) with the token embeddings from the parsed caption through cross-attention ([Decoding](#decoding)).
4. Attend to all the ERP sequences (each is a sequence of text-informed visual tokens) in addition to a learnable regression token and project this to predict the final bounding box ([ERP-Attention](#erpattention)).

### <a name="preprocessing"></a> 1. Preprocessing
The dataset is initially pre-processed to expand captions to [SceneGraphs](https://en.wikipedia.org/wiki/Scene_graph). With `visgator.datasets.refcocog._generator.Generator.generate()`, each image annotation is parsed through a `visgator.utils.graph.SpacySceneGraphParser` ([reference](https://github.com/vacancy/SceneGraphParser)): entities and relationships between them are identified and stored in a graph object. The generator encodes each dataset sample as a tuple of `(image_path, caption, graph)`.

```python -m visgator --phase generate --config config/example.yaml```

The generated annotations file for RefCOCOg in UMD format:
```{
  "test": [
    {
      "image": "COCO_train2014_000000380440.jpg",
      "caption": {
        "sentence": "the man in yellow coat",
        "graph": {
          "entities": [
            {
              "span": "the man",
              "head": "man"
            },
            {
              "span": "yellow coat",
              "head": "coat"
            }
          ],
          "relations": [
            {
              "subject": 0,
              "predicate": "in",
              "object": 1
            }
          ]
        }
      },
      "bbox": [
        374.31,
        65.06,
        136.04,
        201.94
      ]
    },
    ...
  ]
}
```

### <a name="instantiation"></a> 2. Instantiation
Each `(image, caption)` pair is encoded with respectively the vision and the text backbone of CLIP (ViT-B/32). Note that the final projections leading to the common embedding space are discarded (`visgator.models.erpa._model.forward()`). 

In parallel, each `(image, caption)` pair is also processed with [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), implemented with `visgator.models.erpa._detector.Detector`, to generate region proposals for each entity (`visgator.models.erpa._misc.Graph.new()`). 

The per-entity region proposals, the text embeddings and the caption are embedded in a Nested SceneGraph, that is a set of graphs for the training batch. Moreover, the padded region proposals are organized in `visgator.utils.bbox.BBoxes` objects to apply handy geometrical operators.
Finally, a `NestedGraph` object is passed to the ERP-Decoder, along with the formatted bounding boxes and the image embeddings.

### <a name="decoding"></a> 3. Decoding

In each ERP, a gaussian heatmap is computed for each entity bounding box. The union of the heatmaps constitutes the mask for the input image (`visgator.models.erpa.Decoder.forward()`). A stack of attention layers processes the ERPs so that the visual tokens attend the text embeddings, eventually re-arranged in nodes and edges of the NestedGraph. 

### <a name="erpattention"></a> 4. ERP-Attention
The batch NestedGraph is eventually processed by the `visgator.models.erpa._head.RegressionHead`: each ERP token sequence is summed to a positional encoding (sequence-wise). A stack of self-attention layers attends to the sequence of concatenated ERPs, in addition to a learnable token. The latter is eventually projected linearly to predict the target bounding box. 

