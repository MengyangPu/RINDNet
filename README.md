## RINDNet
> [RINDNet: Edge Detection for Discontinuity in Reflectance, Illumination, Normal and Depth](https://arxiv.org/abs/2108.00616)                 
> Mengyang Pu, Yaping Huang, Qingji Guan and Haibin Ling                 
> *ICCV 2021* (oral)

Please refer to [supplementary material](https://pan.baidu.com/s/1oMteiIaPwjWgH-ihCA2S5g) (code:p86d) (~60M) for more results.

### Benchmark --- 🔥🔥BSDS-RIND🔥🔥
BSDS-RIND is the first public benchmark that dedicated to studying simultaneously the four edge types, namely Reflectance Edge (RE), Illumination Edge (IE), Normal Edge (NE) and Depth Edge (DE). It is created by carefully labeling images from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
The datasets can be downloaded from:
- Original images: [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- Our annotations: BSDS-RIND ([BaiDuNetdisk](https://pan.baidu.com/s/1UPIIqhZtQte4RO5WwcjqFg), code:e7rg ; [GoogleDrive](https://drive.google.com/drive/folders/1W4bF2pMEa5g2Wc0qskr-GiotNx0XAbeX?usp=sharing))

<img src="fig/examples.png" width="500">


### Abstract
As a fundamental building block in computer vision, edges can be categorised into four types according to the discontinuity in *surface-Reflectance*, *Illumination*, *surface-Normal* or *Depth*. While great progress has been made in detecting generic or individual types of edges, it remains under-explored to comprehensively study all four edge types together. In this paper, we propose a novel neural network solution, *RINDNet*, to jointly detect all four types of edges. Taking into consideration the distinct attributes of each type of edges and the relationship between them, RINDNet learns effective representations for each of them and works in three stages. In stage I, RINDNet uses a common backbone to extract features shared by all edges. Then in stage II it branches to prepare discriminative features for each edge type by the corresponding decoder. In stage III, an independent decision head for each type aggregates the features from previous stages to predict the initial results. Additionally, an attention module learns attention maps for all types to capture the underlying relations between them, and these maps are combined with initial results to generate the final edge detection results. For training and evaluation, we construct the first public benchmark, BSDS-RIND, with all four types of edges carefully annotated. In our experiments, RINDNet yields promising results in comparison with state-of-the-art methods.

<img src="fig/illustration.png" width=90%>

### Usage
1. Clone this repository to local
```shell
git clone https://github.com/MengyangPu/RINDNet.git
```
2. Download the [augmented data](https://drive.google.com/file/d/1CO5QZvuzD9AoQ-t9pFsRJSL7ldAkOSqc/view?usp=sharing) to the local folder /data

3. Download Pre-trained model

|   Method   | model                       | Pre-trained Model           |
| ---------- | --------------------------- | --------------------------- | 
| HED        |[model](modeling/hed.py)     | [download]() |
| CED        |[code](https://github.com/Wangyupei/CED)    | [download]() |
| RCF        |[model](modeling/rcf.py)     | [download]() |
| BDCN       |[model](modeling/bdcn.py)    | [download]() |
| DexiNed    |[model](modeling/dexined.py) | [download]() |
| CASENet    |[model](modeling/casenet.py) | [download]() |
| DFF        |[model](modeling/dff.py)     | [download]() |
|\*DeepLabv3+|[model](modeling/deeplab.py) | [download]() |
|\*DOOBNet   |[model](modeling/doobnet.py) | [download]() |
|\*OFNet     |[model](modeling/ofnet.py)   | [download]() |
| DeepLab    |[model](modeling/deeplab2.py)| [download]() |
| DOOBNet    |[model](modeling/doobnet2.py)| [download]() |
| OFNet      |[model](modeling/ofnet2.py)  | [download]() |
| RINDNet    |[model](modeling/rindnet.py) | [download]() |


### Main results

#### BSDS-RIND

|   Method   | model                       | ODS  | OIS  | AP   | ODS  | OIS  | AP   | ODS  | OIS  | AP   | ODS  | OIS  | AP   | ODS  | OIS  | AP   |
| ---------- | --------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| HED        |[model](modeling/hed.py)     | 0.412| 0.466| 0.343| 0.256| 0.290| 0.167| 0.457| 0.505| 0.395| 0.644| 0.679| 0.667| 0.442| 0.485| 0.393|
| CED        |-                            | 0.429| 0.473| 0.361| 0.228| 0.286| 0.118| 0.463| 0.501| 0.372| 0.626| 0.655| 0.620| 0.437| 0.479| 0.368|
| RCF        |[model](modeling/rcf.py)     | 0.429| 0.448| 0.351| 0.257| 0.283| 0.173| 0.444| 0.503| 0.362| 0.648| 0.679| 0.659| 0.445| 0.478| 0.386|
| BDCN       |[model](modeling/bdcn.py)    | 0.358| 0.458| 0.252| 0.151| 0.219| 0.078| 0.427| 0.484| 0.334| 0.628| 0.661| 0.581| 0.391| 0.456| 0.311|
| DexiNed    |[model](modeling/dexined.py) | 0.402| 0.454| 0.315| 0.157| 0.199| 0.082| 0.444| 0.486| 0.364| 0.637| 0.673| 0.645| 0.410| 0.453| 0.352|
| CASENet    |[model](modeling/casenet.py) | 0.384| 0.439| 0.275| 0.230| 0.273| 0.119| 0.434| 0.477| 0.327| 0.621| 0.651| 0.574| 0.417| 0.460| 0.324|
| DFF        |[model](modeling/dff.py)     | 0.447| 0.495| 0.324| 0.290| 0.337| 0.151| 0.479| 0.512| 0.352| 0.674| 0.699| 0.626| 0.473| 0.511| 0.363|
|\*DeepLabv3+|[model](modeling/deeplab.py) | 0.297| 0.338| 0.165| 0.103| 0.150| 0.049| 0.366| 0.398| 0.232| 0.535| 0.579| 0.449| 0.325| 0.366| 0.224|
|\*DOOBNet   |[model](modeling/doobnet.py) | 0.431| 0.489| 0.370| 0.143| 0.210| 0.069| 0.442| 0.490| 0.339| 0.658| 0.689| 0.662| 0.419| 0.470| 0.360|
|\*OFNet     |[model](modeling/ofnet.py)   | 0.446| 0.483| 0.375| 0.147| 0.207| 0.071| 0.439| 0.478| 0.325| 0.656| 0.683| 0.668| 0.422| 0.463| 0.360|
| DeepLab    |[model](modeling/deeplab2.py)| 0.444| 0.487| 0.356| 0.241| 0.291| 0.148| 0.456| 0.495| 0.368| 0.644| 0.671| 0.617| 0.446| 0.486| 0.372|
| DOOBNet    |[model](modeling/doobnet2.py)| 0.446| 0.503| 0.355| 0.228| 0.272| 0.132| 0.465| 0.499| 0.373| 0.661| 0.691| 0.643| 0.450| 0.491| 0.376|
| OFNet      |[model](modeling/ofnet2.py)  | 0.437| 0.483| 0.351| 0.247| 0.277| 0.150| 0.468| 0.498| 0.382| 0.661| 0.687| 0.637| 0.453| 0.486| 0.380|
| RINDNet    |[model](modeling/rindnet.py) | 0.478| 0.521| 0.414| 0.280| 0.337| 0.168| 0.489| 0.522| 0.440| 0.697| 0.724| 0.705| 0.486| 0.526| 0.432|


### Acknowledgments
- The work is partially done while Mengyang was at Stony Brook University.
- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
- Thanks to previous open-sourced repo:<br/>
  [HED-pytorch](https://github.com/xwjabc/hed)<br/>
  [RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)<br/>
  [BDCN](https://github.com/pkuCactus/BDCN)<br/>
  [DexiNed](https://github.com/xavysp/DexiNed)<br/>
  [DFF](https://github.com/Lavender105/DFF)<br/>
  [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)<br/>
  [DOOBNet-pytorch](https://github.com/yuzhegao/doob)
