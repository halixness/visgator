# Referring Expression Comprehension as Scene Graph Grounding
## Authors
Diego Calanzone, Francesco Gentile
University of Trento
Deep Learning course project, Spring 2023

## Notes on the code
For this project we have decided to develop a complete framework that can be easily extended to many computer vision tasks by simply defining the necessary metrics and inputs and targets. This made extremely easy to test different models without reuqiring any modifications to parts of the code not related to the model.
In this notebook, we do not report all the infrastructure since it is not the relevant part of the project. We only report those code snippets that are associated to our proposed model. The complete framework can be found at https://github.com/FrancescoGentile/visgator in the deepsight branch.

A full code overview with runnable experiments can be found in the [report](https://github.com/halixness/visgator/blob/main/report.ipynb).

## Introduction
Referring Expression Comprehension (REC) is the task of localizing a target object in an image given a natural language expression that refers to it. Most recent approaches (Zhang et al. 2022, Xu et al. 2023, Liu et al. 2023) that obtain state-of-the-art results on this task are not specifically designed for it, but they are designed to solve a large variety of tasks that require fusing vision and language modalities, like open-set object detection, image captioning, visual question answering and so on. In particular, most of these first independently encode the the visual and textual input using vision and text encoders (based on the Transformer architecture) respectively, then another transformer module is used to fuse the two modalities by making the visual features attend to the textual features and vice versa. Finally, the fused features are given in input to another module (a simple head, a transformer decoder, etc.) based on the task that is being solved.
Here we argue that the task of REC requires an high-level understanding of the scene described the region caption. For example, given a caption like "The girl approaching the table while holding a glass", to correctly localize the associated bounding box, we need to first identify all the entities referred by the sentence (the girl, the table, a glass) and the relation that exist among them ((the girl -- approaching -> the table), (the girl -- holding --> a glass)). In other words, we need to extract from the sequence of words that form the sentence an intermediate higher-level representation of the scene. Then, instead of grounding the sequence of words to the image, we can ground the intermediate representation to the image. On the other hand, previously cited approaches, since they need to generalize to many image-text tasks, simply ground the word features (here we make the simplifying assumption that each token correspond to a word) extracted by the text encoder to the image features. Thus, to perform well in such task, the text encoder need to encode into each token not only the meaning of the corresponding word but also its relations with the other entities that in the sentence may be refered by group of tokens. In toher words, the text encoder need to learn to extract from the sequence of tokens a higher-level representation without being explicitly supervised to do so.
Based on this observation, we propose a new approach to REC that is specifically designed for this task by making the network directly exploit the higher-level semantic information encoded in the input sentence. In particular, from the input sentence we extract a scene graph representing which entities are present in the region and how they are related to each other. Then, we localize the target region by localizing in the image the referred entities that satisfy the referred relations.

## High level architecture overview
<img src="https://camo.githubusercontent.com/456e9576e3742224dd81b8919137fc51d9b770938a824cdde0ac6fd61ea38c1b/68747470733a2f2f6769746875622e636f6d2f4672616e636573636f47656e74696c652f7669736761746f722f626c6f622f6465657073696768742f646f63732f696d672f7367672e706e673f7261773d74727565" />
Our architecture is highly inspired to DETR-like models (Carion et al. 2020, Gao et al. 2021, Liu et al. 2022). Such models use a (CNN) backbone followed by a transformer-based vision encoder to extract visual features from the input image. Then a set of queries (representing candidate bounding boxes) is given in input to a transformer-based decoder, where such queries go through layers of self-attention and cross-attention with the visual features. Finally, the output of the decoder is given in input to a simple head that predicts the bounding boxes coordinates and the class of each bounding box. At training time, Hungarian matching is used to obtain a one-to-one matching between a query and a ground truth bounding box. Once such association is obtained, the loss is computed by comparing the predicted bounding box with the associated ground truth bounding box. At inference time, the predicted bounding boxes are filtered by a simple post-processing step to remove the predicted bounding boxes that have a low confidence score.
Similarly, we extract the visual features by employing a transformer-based vision encoder (no backbone is used since we use the CLIP vision encoder). Then, differently from DETR-like models, we do not generate a predefined set of fixed or learnable queries, but we create a graph based on the one extracted from the sentence. In particular, for each entity in the sentence scene graph we create multiple nodes in the graph (since in the image there may be multiple instances of the same entity) whose embeddings are initialized with the embedding obtained by giving in input to the CLIP text encoder the entity textual description extracted from the sentence. Then, we create an edge between two nodes if the corresponding entities are related in the sentence scene graph; as before, the edge features are initialized by encoding the textual description of the relation with the CLIP text encoder. Then, the generated graph is given in input to the transformer-based decoder whose blocks consist of a sequence of Multi-Head Cross Attention, Graph Attention and FFN.
In the multi-head cross attention layer, each query (nodes + edges) can attend to the visual features extracted from the vision encoder. This allow each query to verify whether the associated entity/relation is present in a specific region of the image. Then, in the graph attention layer, each node can communicate with its neighbours and the associated relations to verify whether the encoded instance of the entity satisfy the relations encoded in the sentence scene graph.
Finally, from the graph outputted by the decoder, we extract the nodes that represent the subject of the sentence (i.e. the target entity) and we give them in input to a simple head to obtain candidate bounding boxes for the target entity. At training time, a simple matching algorithm is applied to associate the ground truth bounding box with one of the predicted bounding boxes and the loss is computed. At inference time, we select the node (and the obtained bounding box) whose embedding is the most similar to the embedding obtained by giving in input to the CLIP text encoder the full sentence.
As currently presented, the decoder should also perform open-set object detection, since for each entity it should localize all the instances in the image. Thus, we should create a sufficient high number of nodes for each entity to be able to localize all the instances of the entity in the image. For example, Grounding DINO (Liu et al. 2023) creates 900 queries for each image. This would clearly require a lot of memory and computation power. Furthermore, current open-set object detectors are trained on huge amounts of data, on many GPUs and for long period of times (Grounding DINO uses 64 A100). Given the limited resources availables, we decided to employ an open-set object detector to obtain all instances of an entity in the image and an estimate of their location. In this way the decoder does not need to perform open-set object detection from scratch but it only needs to refine the estimated locations. Since existing open-set object detectors are mainly trained on closed object detection datasets, where each entity to be detected is represented by a single noun (i.e., the category name), to make the detector localize an entity we do not use its full textual description. Instead, for each entity we extract a single noun that best describe that entity. For example, given the entity The woman with dark hair, we extract the noun woman to localize the entity.