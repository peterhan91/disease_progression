# Predicting Osteoarthritis Progression via Unsupervised Adversarial Representation Learning
(c) Tianyu Han and Daniel Truhn, RWTH Aachen University, 2021

## About
### What's included in this Repo
The repository includes the codes for data / label preparation and inferencing the future knee radiograph, training and testing the baseline classifier and also the links to the pre-trained generative model.

### Focus of the current work
Osteoarthritis (OA) is the most common joint disorder in the world affecting 10% of men and 18% of women over 60 years of age. In this paper, we present an unsupervised learning scheme to predict the future image appearance of patients at recurring visits. 
<center>
<img src="pics/result.png" width="800"/> 
</center>

By exploring the latent temporal trajectory based on knee radiographs, our system predicts the risk of accelerated progression towards OA and surpasses its supervised counterpart. We demonstrate this paradigm with seven radiologists who were tasked to predict which patients will undergo a rapid progression.
<center>
<img src="pics/result_auc.png" width="800"/> 
</center>


### Requirements
```
pytorch 1.8.1
tensorboard 2.5.0
numpy 1.20.3
scipy 1.6.2
scikit-image 0.18.1
pandas
tqdm
glob
pickle5
```
- **[StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)**   
    This repository is an official reimplementation of StyleGAN2-ADA in PyTorch, focusing on correctness, performance, and compatibility. 
- **[KNEE Localization](https://github.com/MIPT-Oulu/KNEEL)** \
    The repository includes the codes for training and testing, annotations for the OAI dataset and also the links to the pre-trained models.
- **[Robust ResNet classifier](https://github.com/peterhan91/Medical-Robust-Training)** \
    The repository contains codes for developing robust ResNet classifier with a superior performance and interpretability. 


## How to predict the future state of a knee
### Preparing the training data and labels
Download all available OAI and MOST images from https://nda.nih.gov/oai/ and https://most.ucsf.edu/. The access to the images is free and painless. You just need to register and provide the information about yourself and agree with the terms of data use. Besides, please also download the label files named  ```Semi-Quant_Scoring_SAS```  and ``` MOSTV01235XRAY.txt ``` from [OAI](https://nda.nih.gov/oai/full_downloads.html) and [MOST](https://most.ucsf.edu/), separately. 

Following the repo of [KNEE Localization](https://github.com/MIPT-Oulu/KNEEL), we utilized a [pre-trained Hourglass network](http://mipt-ml.oulu.fi/models/KNEEL/snapshots_release.tar.xz) and extracted 52,981 and 20,158 (separated left or right) knee ROI (256x256) radiographs from both OAI and MOST datasets. We further extract the semi-quantitative assessment Kellgren-Lawrence Score (KLS) from the labels files above. To better relate imaging and tabular data together, in OAI dataset, we name the knee radiographs using ```ID_BARCDBU_DATE_SIDE.png```, e.g., ```9927360_02160601_20070629_l.png```. For instance, to generate the KLS label file (`most.csv`) of the MOST dataset, one can run:
```.bash
python kls.py
```

### Training a StyleGAN2 model on radiological data
Follow the official repo [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch), datasets are stored as uncompressed ZIP archives containing uncompressed PNG files.
Our datasets can be created from a folder containing radiograph images; see [`python dataset_tool.py --help`](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/docs/dataset-tool-help.txt) for more information. 
In the auto configuration, training a OAI GAN boils down to:
```.bash
python train.py --outdir=~/training-runs --data=~/OAI_data.zip --gpus=2
```
The total training time on 2 Titan RTX cards with a resolution of 256x256 takes around 4 days to finish. The best GAN model of our experiment can be downloaded at [here](https://drive.google.com/file/d/1spL4cDH6B8rL1UN2Nx3gF9TnoLiBK-gU/view?usp=sharing).

### Projecting training radiographs to latent space 
To find the matching latent vector for a given training set, run:
```.bash
python projector.py --outdir=~/pro_out --target=~/training_set/ --network=checkpoint.pkl
```
The function `multi_projection()` within the script will generate a dictionary contains pairs of image name and its corresponding latent code and individual projection folders. 

### Synthesize future radiograph
- **require**: A pre-trained network G, test dataframe path (contains test file names), and individual projection folders (OAI training set).
To predict the baseline radiographs within the test dataframe, just run: 
```.bash
python prog_w.py --network=checkpoint.pkl --frame=test.csv --pfolder=~/pro_out/ 
```

## Estimating the risk of OA progression
In this study, we have the ability to predict the morphological appearance of the radiograph at a future time point and compute the risk based on the above synthesized state. We used an adversarially trained [ResNet model](https://github.com/peterhan91/Medical-Robust-Training) that can correctly classify the KLS of the input knee radiograph.

We denote $x_i$, $x_j$ as baseline and follow-up knee radiographs respectively, and  $c_i$, $c_j$ as KLS classes for $x_i$ and $x_j$ ranging from 0 to 4.
We defined patients with imminent OA onset and/or progression towards osteoarthritic changes as those, who showed progress of more than one KLS ($c_j - c_i > 1$) over scans.
We then computed the probability of OA progression ($y=1$) as a sum of joint probabilities $\{ (c_i,\, c_j) \}$ which fulfill the condition that the KLS change is larger than 1, i.e., $c_j - c_i > 1$, between its prior and follow-up visit radiographs:
$p(y=1|x_i, x_j) = \sum_{\{ (c_j-c_i > 1) \}}p(\text{KLS}=c_i|x_i)\times p(\text{KLS}=c_j|x_j)$.

To generate the ROC curve of our model, run: 
```.bash
python risk.py --ytrue=~/y_true.npy --ystd=~/baseline/pred/y_pred.npy --ybase=~/kls_cls/pred/ypred.npy --yfinal=~/kls_cls/pred/ypred_.npy --df=~/oai.csv
```
### 

## Baseline classifier
To compare what is achievable with supervised learning based on the existing dataset, we finetune a ResNet-50 classifier pretrained on ImageNet that tries to distinguish fast progressors based on baseline radiographs in a supervised end-to-end manner.
The output probability of such a classifier is based on baseline radiographs only. 
To train the classifier, after putting the label files to the `base_classifier/label` folder, one can run:
```.bash
cd base_classifier/
python train.py --todo train --data_root ../Xray/dataset_oai/imgs/ --affix std --pretrain True --batch_size 32
```
To test, just run: 
```.bash
cd base_classifier/
python train.py --todo test --data_root ../Xray/dataset_oai/imgs/ --batch_size 1
```


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

```
```

## Acknowledgments

- **KNEE Localization** (https://github.com/MIPT-Oulu/KNEEL)
- **StyleGAN2-ADA-Pytorch** (https://github.com/NVlabs/stylegan2-ada-pytorch)