# CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection

The official implementation of CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection (ICML2024 [CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection (openreview.net)](https://openreview.net/pdf?id=xFDJBzPhci)).

## Abstract

Recent vision-language pre-trained models (VL-PTMs) have shown remarkable success in open-vocabulary tasks. However, downstream use cases often involve further fine-tuning of VL-PTMs, which may distort their general knowledge and impair their ability to handle distribution shifts. In real-world scenarios, machine learning systems inevitably encounter both covariate shifts (e.g., changes in image styles) and semantic shifts (e.g., test-time unseen classes). This highlights the importance of enhancing out-of-distribution (OOD) generalization on covariate shifts and simultaneously detecting semantic-shifted unseen classes. Thus a critical but underexplored question arises: How to improve VL-PTMs' generalization ability to closed-set OOD data, while effectively detecting open-set unseen classes during fine-tuning? In this paper, we propose a novel objective function of OOD detection that also serves to improve OOD generalization. We show that minimizing the gradient magnitude of energy scores on training data leads to domain-consistent Hessians of classification loss, a strong indicator for OOD generalization revealed by theoretical analysis. Based on this finding, we have developed a unified fine-tuning framework that allows for concurrent optimization of both tasks. Extensive experiments have demonstrated the superiority of our method.

## Pipeline

**Overview of the CRoFT framework**

![pipeline-croft](/Users/zl/Desktop/pipeline-croft.png)

## How to Install

This code is built on top of the awesome [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and [CoOp]([KaiyangZhou/CoOp: Prompt Learning **for** Vision-Language Models (IJCV'22, CVPR'22) (github.com)](https://github.com/KaiyangZhou/CoOp))., run `pip install -r requirements.txt` under `CRoFT/CoOp/` to install the required packages.

```shell
git clone https://github.com/LinLLLL/CRoFT.git
cd CRoFT/CoOp

conda create -n CRoFT python=3.9
conda activate CRoFT

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

## Datasets

Follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to install **ImageNet, ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R**, and **other 10 datasets** referring to CoOp.

For the OOD datasets, such as **PACS** and **VLCS**, are publicly available but need to be downloaded manually. Please refer this [instruction]([OoD-Bench/data/README.md at main · m-Just/OoD-Bench (github.com)](https://github.com/m-Just/OoD-Bench/blob/main/data/README.md)) for OOD datasets download. Please make sure that the directory structure of each dataset is arranged as follows:

**PACS**

```
PACS
├── images
    ├── art_painting
		├── cartoon
		├── photo
		└── sketch
├── test_on_art_painting.json
├── test_on_cartoon.json
├── test_on_photo.json
└── test_on_sketch.json
```

**VLCS**

```
VLCS
├── images
    ├── CALTECH
		├── LABELME
		├── PASCAL
		└── SUN
├── test_on_caltech.json
├── test_on_labelme.json
├── test_on_pascal.json
└── test_on_sun.json
```

## ▶️ ▶️ ▶️ How to Run ▶️ ▶️ ▶️

We provide the running scripts in `CoOp/scripts`.  We take CRoFT as an example, other methods can be similarly evaluated. Make sure you change the path on  `DATA` in shell files  under `CoOp/scripts/CRoFT` and run the commands under `CoOp/scripts/CRoFT`. 

###  ------------------------------   SETUP-I ------------------------------ 

#### For training CRoFT on the in-distribution ImageNet datasets:

```
python run_setup1.py
```

#### For evaluating CRoFT on the closed-set OOD datasets and open-set OOD datasets:

```
python test_setup1.py
```

#### For loading energy distribution of different types of datasets:

```
bash test_energy.sh
```

### ------------------------------   SETUP-II ------------------------------ 

#### For training CRoFT on the in-distribution PACS or VLCS datasets:

```
python run_setup2.py
```

#### After `run_setup2.py`, evaluation on the closed-set OOD datasets is also completed.

#### For evaluating CRoFT on the open-set OOD datasets:

```
python test_setup2_energy.py
```

## ▶️ ▶️ ▶️ Collect Results ▶️ ▶️ ▶️

#### For collecting CRoFT's OOD generalization results in SETUP-I:

```shell
# run the commands under CoOp/
python collect_result_set1_oodg.py
```

#### For collecting CRoFT's OOD detection results in SETUP-I:

```shell
# run the commands under CoOp/
python collect_result_set1_oodd.py
```

#### For collecting CRoFT's OOD detection results in SETUP-II:

```shell
# run the commands under CoOp/
python collect_result_set2_oodg.py
```

#### For collecting CRoFT's OOD detection results in SETUP-II:

We probide two OOD detection methods in SETUP-II, i.e., inferring energy score and KNN distance.  Make sure you have completed the evluation process of `python test_setup2_energy.py` before you run `python test_setup2_knn.py`.

```shell
# run the commands under CoOp/
# inferring energy score
python collect_result_set2_oodg.py
# inferring KNN distance:
python test_setup2_knn.py
```

The evaluation results are then saved to the folders `output` and `eval_open_ood` or displayed directly on your screen.

## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) [CoCoOp]([KaiyangZhou/CoOp: Prompt Learning for Vision-Language Models (IJCV'22, CVPR'22) (github.com)](https://github.com/KaiyangZhou/CoOp)), [Tip-Adapter-F]([gaopengcuhk/Tip-Adapter (github.com)](https://github.com/gaopengcuhk/Tip-Adapter)), [DPLCLIP]([shogi880/DPLCLIP (github.com)](https://github.com/shogi880/DPLCLIP)), [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter) and the OOD generalization benchmark [OoD-Bench]([ynysjtu/ood_bench (github.com)](https://github.com/ynysjtu/ood_bench)). Thanks for their wonderful works.

## Citation

If you use this code in your research, please kindly cite this paper:

```
@article{zhu2024croft,
  title={CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection},
  author={Zhu, Lin and Yang, Yifeng and Gu, Qinying and Wang, Xinbing and Zhou, Chenghu and Ye, Nanyang},
  journal={arXiv preprint arXiv:2405.16417},
  year={2024}
} 
```

## Contact ✉️

If you have any question about this project, please feel free to contact zhulin_sjtu@sjtu.edu.cn.