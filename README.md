## RINDNet
> [RINDNet: Edge Detection for Discontinuity in Reflectance, Illumination, Normal and Depth](https://arxiv.org/abs/2108.00616)                 
> Mengyang Pu, Yaping Huang, Qingji Guan and Haibin Ling                 
> *ICCV 2021* (oral)

Please refer to [supplementary material](https://pan.baidu.com/s/1oMteiIaPwjWgH-ihCA2S5g) (code:p86d) (~60M) for more results.

### Abstract
As a fundamental building block in computer vision, edges can be categorised into four types according to the discontinuity in *surface-Reflectance*, *Illumination*, *surface-Normal* or *Depth*. While great progress has been made in detecting generic or individual types of edges, it remains under-explored to comprehensively study all four edge types together. In this paper, we propose a novel neural network solution, *RINDNet*, to jointly detect all four types of edges. Taking into consideration the distinct attributes of each type of edges and the relationship between them, RINDNet learns effective representations for each of them and works in three stages. In stage I, RINDNet uses a common backbone to extract features shared by all edges. Then in stage II it branches to prepare discriminative features for each edge type by the corresponding decoder. In stage III, an independent decision head for each type aggregates the features from previous stages to predict the initial results. Additionally, an attention module learns attention maps for all types to capture the underlying relations between them, and these maps are combined with initial results to generate the final edge detection results. For training and evaluation, we construct the first public benchmark, BSDS-RIND, with all four types of edges carefully annotated. In our experiments, RINDNet yields promising results in comparison with state-of-the-art methods.

<img src="fig/illustration.png" width=90%>

### Benchmark --- 🔥🔥BSDS-RIND🔥🔥
BSDS-RIND is the first public benchmark that dedicated to studying simultaneously the four edge types, namely Reflectance Edge (RE), Illumination Edge (IE), Normal Edge (NE) and Depth Edge (DE). It is created by carefully labeling images from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
The datasets can be downloaded from:
- Original images: [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- Our annotations: BSDS-RIND ([BaiDuNetdisk](https://pan.baidu.com/s/1UPIIqhZtQte4RO5WwcjqFg), code:e7rg ; [GoogleDrive](https://drive.google.com/drive/folders/1W4bF2pMEa5g2Wc0qskr-GiotNx0XAbeX?usp=sharing))

<img src="fig/examples.png" width="500">


### Code and Main results ----- Coming Soon...

### Citation
If you use the dataset or this code, please consider citing our work
```bibtex
@inproceedings{pu2021RINDNet,
    title={RINDNet: Edge Detection for Discontinuity in Reflectance, Illumination, Normal and Depth}, 
    author={Mengyang Pu, Yaping Huang, Qingji Guan and Haibin Ling},
    booktitle={ICCV},
    year={2021}
}
```

### Acknowledgments
- The work is partially done while Mengyang was at Stony Brook University.
- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
