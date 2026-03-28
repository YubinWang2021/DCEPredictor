# A Codebase for Training Dense Correspondence Embedding Predictor

<span style="font-size:20px;">[Under Review at IEEE TCSVT]</span>

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![Linux](https://img.shields.io/badge/OS-Linux-FCC624?logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows&logoColor=white) ![Under Review](https://img.shields.io/badge/Status-Under%20Review-yellow)

This repository contains the training data for a dense correspondence embedding predictor. In this codebase, we employ datasets with **ground truth annotations**, such as DensePose-COCO, and present an alternative approach for training dense correspondence embeddings based on ground truth 2D-3D annotations. You may train this predictor using either the implementation provided in this repository, or the one from the codebase of **Structured Distilled 3D Gait Fields (SD-3DGF)**.

![image-20260328163202037](figs/embed_True.png)

## 1. News

- [x] Provide all required training data
- [x] Release Pytorch Implementation Code
- [x] Release Visualization Code
- [ ] Code cleanup and improvements based on feedback
## 1. Features

#### Supported CNN backbones
- `effunet`: EfficientUNet
- `unet`: UNet (Standard Version)
- `darknet`: DarkNet-v2
#### Supported ViT backbones
- `dinov2`: DINOv2
- `clip`:  CLIP (ViT Image Encoder)

## 2. Data Preparation
#### First Clone the repository
```bash
git clone https://github.com/YubinWang2021/DCEPredictor
```
#### 2.1 Download the DensePose-COCO 2014 datasets 

![image-20260328163202037](figs/Fig3.png)

Create a folder named `data`  and a folder named `checkpoint` inside the repository, 

**2.1.1 For the raw dataset:** 

(1) [recommended] You can use our processed data for VCCRe-ID datasets 

(2) [Alternative] You can process it by yourself by running the following command line (**Note**: replace the path to the folder storing the datasets and the dataset name).

```bash
python datasets/prepare.py --root "your VCCRe-ID Dataset Root" --dataset_name vccr
```
This will generate train.pkl, query.pkl, and gallery.pkl in [dataset_name]. 

**2.1.2 For the dense 2D-3D paired data:** 

(1) [recommended] You can use our processed data for VCCRe-ID datasets.

(2) [Alternative] You can use our data generation pipeline to generate dense 2D-3D paired data for your customized data.

Pytorch Implementation: https://github.com/YubinWang2021/Dense-Paired-Data-Generation 

#### 2.2 Download our pretrained clothes classifier.
| Dataset | Pre-Trained Clothes Classifier|
|:----------:|:----------:|
| CCV-S | [link](https://drive.google.com/file/d/1eXgXnTofXnutDnbCpHszTnJceLwJMZzc/view?usp=sharing) |
| CCV-R | [link](https://drive.google.com/file/d/1Ak5V_u6YmEjjq1wblpttYXEYmVcvB3Wq/view?usp=sharing) |
| CCVID | [link](https://drive.google.com/file/d/1F25i-KjBz3qOIWSFpJQ344UyzQs8pFTL/view?usp=sharing) |
| VCCR | [link](https://drive.google.com/file/d/1klPyXXUjF8VsnUrOCv-ncSS8c1HYmQlU/view?usp=sharing) |

#### 2.3 Download our pretrained 3D vertex embeddings.

![image-20260328163202037](figs/Fig2.png)

Download the pretrained 3D vertex embeddings at  [this link](https://drive.google.com/file/d/1SSYH3zO-ax-aKRHLoaGRsxBQXjXvwAb4/view?usp=sharing).
#### 2.4  Download our pretrained dense embedding correspondence predictor checkpoint or train your own  dense correspondence embedding predictor.
(1) Download our pretrained checkpoint: [this link](https://drive.google.com/file/d/1CbE1WelNSUnP7VCBofmGcHBgbczyruBw/view?usp=sharing). 

(2) For your own customized data, you can refer to our pytorch implementation: https://github.com/YubinWang2021/DCEPredictor  

#### 2.5 Please extract the compressed pretrained data package and organize it according to the directory structure provided below.
**(1) Data Folder:**

```text
Data/
├── VCCR/                  # VCCR dataset
│   ├── dense_corr/        # Dense 2D-3D correspondence
│   ├── mask/              # Segmentation masks
│   ├── train.pkl           # Training split
│   ├── query.pkl           # Query split
│   └── gallery.pkl         # Gallery split
├── CCVID/                 # CCVID dataset
│   ├── dense_corr/
│   ├── mask/
│   ├── train.pkl
│   ├── query.pkl
│   └── gallery.pkl
├── CCV-R/                 # CCV-R dataset
│   ├── dense_corr/
│   ├── mask/
│   ├── train.pkl
│   ├── query.pkl
│   └── gallery.pkl
└── CCV-S/                 # CCV-S dataset
    ├── dense_corr/
    ├── mask/
    ├── train.pkl
    ├── query.pkl
    └── gallery.pkl
```

**(2) Checkpoint Folder:**

```text
checkpoint/
├── vccr_clothes_classifier.pth
├── ccvid_clothes_classifier.pth
├── ccvr_clothes_classifier.pth
├── ccvs_clothes_classifier.pth
└── dce_pretrained.pth
```

## 2. Running instructions

### Getting started

#### Create virtual environment

First, create a virtual environment for the repository
```bash
conda create -n sd3dgf python=3.10
```
then activate the environment 
```bash
conda activate sd3dgf
```
Next, install the package by running
```bash
python setup.py install
```
Then, install the dependencies
```bash
pip install -r requirements.txt
```

## 3. Configuration Options

Go to `./config.py` to modify configurations accordingly
- Dataset name
- Number of epochs
- Batch size
- Learning rate
- backbone (according to model names above)
- Choice of loss functions
- Hyper-parameters

If training from checkpoint, copy checkpoint path and paste to RESUME in `./config.py`.

## 4. Run Training/Testing Scripts

Create a folder named `work_space` inside the repository, then create two subfolders named `save` and `output`.

```
data
work_space
|--- save
|--- output
main.sh
```
#### Run

```bash
bash main.sh
```

or

```bash
python train.py
python test.py
```

Trained model will be automatically saved to `work_space/save`.
Testing results will be automatically saved to `work_space/output`.

## 5. Visualization

![image-20260328163202037](figs/Fig4.png)

For 3D visualization related code, please refer to the visualization code in our dedicated repository:  
https://github.com/YubinWang2021/DCEPredictor  

We will organize and release the full 3D visualization toolkit as a standalone repository here in subsequent updates.

## Citation

The paper is under review at IEEE TCSVT.
- Note
  - All pedestrian video/image data are for research purposes only and must comply with the usage policies of the corresponding open-source datasets.
  
## Acknowledgement
Related Repositories:
-  [CAL](https://github.com/guxinqian/Simple-CCReID). 
-  [VCCReID-Baseline](https://github.com/dustin-nguyen-qil/VCCReID-Baseline). 







