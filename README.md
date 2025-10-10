# Raster-to-Graph
By Sizhe Hu, Wenming Wu, Ruolin Su, Wanni Hou, Liping Zheng and Benzhu Xu.

This repository is an official implementation of the paper [Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer](https://wutomwu.github.io/publications/2024-Raster2Graph/paper.pdf). (EG 2024)

## Introduction
Raster-to-Graph is a novel automatic recognition framework, which achieves structural and semantic recognition of floorplans, addresses the problem of obtaining high-quality vectorized floorplans from rasterized images. 

We represent vectorized floorplans as structural graphs embedded with floorplan semantics, thus transforming the floorplan recognition task into a structural graph prediction problem. We design an autoregressive prediction framework using the neural network architecture of the visual attention Transformer, iteratively predicting the wall junctions and wall segments of floorplans in the order of graph traversal. Additionally, we propose a large-scale floorplan dataset containing over 10,000 real-world residential floorplans. Extensive experiments demonstrate the effectiveness of our framework, showing significant improvements on all metrics. Qualitative and quantitative evaluations indicate that our framework outperforms existing state-of-the-art methods. 

We contribute the following: 

• An automatic recognition framework to obtain high-quality vectorized floorplans from rasterized images through one neural network. 

• A novel autoregressive model that iteratively predicts structures and semantics of floorplans in the order of graph traversal. 

• A large-scale floorplan dataset containing more than 10,000 realistic residential floorplans with dense annotations both on structures and semantics. To the best of our knowledge, this is currently the largest dataset available for floorplan recognition. The dataset has much potential to inspire more research. 

To learn more, please refer to our [paper](https://wutomwu.github.io/publications/2024-Raster2Graph/paper.pdf).

# Test results on different machines & environments:
![image](https://github.com/user-attachments/assets/6a23fd3c-e902-4936-a3c4-7da6636ddfe1)

![image](https://github.com/user-attachments/assets/470ed807-4db5-4a83-8004-cca156c619bd)

The difference of about 0.1 in the Edge F-1 is because the 96.1 in the paper and the 96.2 here use different precision calculations (preserving 1 decimal place for Prec and Recall before calculation, and directly using the high-precision data inside the program)

## Environments
Our repo was developed and tested with Python 3.7, cuda 11.1, Windows 10.

To install the necessary dependencies:

```bash
pip install -r requirements.txt
```

or this:
```bash
conda create -n R2G python=3.7.13

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install cython==0.29.30 imageio==2.22.0 libpng==1.6.37 matplotlib==3.5.2 networkx==2.6.3 numpy==1.21.6 opencv-python==4.5.3.56 pillow==9.1.1 scikit-image==0.19.2 scipy==1.7.3 shapely==2.0.1 tqdm==4.64.0
```

To install the necessary deformable attention module provided by [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR):

```bash
cd models/ops/
sh make.sh
cd ../..
```

(If you would like to know the version information of all packages, please refer to the `requirements_full.txt`.)

## Data
### Note: Applications for relevant data will be processed uniformly every Sunday evening. 
### 
### Some researchers reported that they were unable to receive the email. If you haven't received it by Monday morning (UTC+8), please check your spam folder. If you really haven't received it, please contact me: 2024010072@mail.hfut.edu.cn. I will send you a new one. 

Our dataset includes:

(I) Centered 512*512 images obtained by processing "LIFULL HOME'S High Resolution Floor Plan Image Data" ("LIFULL HOME'S Data" for short). 

We use centered 512*512 images (instead of "LIFULL HOME'S Data") in our model, because "LIFULL HOME'S Data" has considerable distinctions in sizes and margins.

(II) annotations. 

Here is a detailed guide on how to get the data:

### Step 1: Access the "LIFULL HOME'S Data"

We do not have the permission to share the "LIFULL HOME'S Data". To access it, you need to apply through the following link: [LIFULL HOME'S High Resolution Floor Plan Image Data](https://www.nii.ac.jp/dsc/idr/en/lifull/). Upon approval, you will receive the images. 

Their data contains from "photo-rent-madori-full-00" to "photo-rent-madori-full-0f", we only use "photo-rent-madori-full-00", so you do not need to download from "photo-rent-madori-full-01" to "photo-rent-madori-full-0f". 

Now you should have a folder named "photo-rent-madori-full-00" that contains approximately 300000 images. You can place this folder at any path of your choice.

### Step 2: Access the Annotations

The annotations can be obtained from [Raster-to-Graph Dataset](https://docs.google.com/forms/d/e/1FAIpQLSexqNMjyvPMtPMPN7bSh_1u4Q27LZAT-S9lR_gpipNIMKV5lw/viewform). 

The download includes 3 folders:

`annot_npy`, `annot_json`: our annotations.

`original_vector_boundary`: boundary boxes of "LIFULL HOME'S Data". We use these data to process "LIFULL HOME'S Data" to centered 512*512 images. (Not used in our model.)

You simply need to place these 3 folders under the `data` folder.

### Step 3: Process the "LIFULL HOME'S Data" to Centered 512x512 Images

**Note: You need to use pillow=8.0.0 to run this script. Although `requirements.txt` specifies pillow=9.1.1, `image_process.py` was developed and tested with pillow=8.0.0. If you use pillow=9.1.1, the resulting processed images may have slight differences, but it shouldn't make a significant impact on the final testing results.**

You should have gotten the "LIFULL HOME'S Data" in Step 1. 

The 512*512 images are generated by running the `image_process.py` in `data`. Change the "original_images_path" variable to the path where you have placed the "photo-rent-madori-full-00" at, and just run the `image_process.py`.

After successfully running this script, the `test`, `train`, and `val` folders will contain 500, 9804, and 500 .jpg images, respectively, which will serve as the input for the model.

### Step 4: Confirm the Data You Have Now

Your `data` directory should now contain image data (`test`, `train`, `val` each with 500, 9804, 500 images respectively) and annotation data (`annot_npy` with 10804 .npy files and `annot_json` with 3 .json files).

## Usage
Here is a [link](https://drive.google.com/file/d/1A_pyF0KHo-fja-fNqkoFkMUxo0W7GHhC/view) to our trained model. To test on this model, please straightly run the following command:

```
python test.py
```

If you would like train your model, please adjust the arguments in `args.py` and then run:

```
python train.py
```
**Note, at the 80th epoch, you should terminate the training process, modify: 
(1) the lr=2e-4 lr_backbone=2e-5 lr_linear_proj=2e-5 to (2e-5 2e-6 2e-6) manually in `args.py`, 
(2) the 'resume' in `args.py` to 'path/to/the/79th_checkpoint'
and rerun**

To run the demo, you can prepare your own image folder (which can contain just 1 image), and then modify the `demo.py` file to change the input parameter of the `MyDataset_demo`. You can then run the following command:

```
python demo.py
```

The visualized output will be located in `output/your_path_name`. Note that the image data we used for training has been processed as described in the Data section earlier. If your image data is processed differently, you might need to retrain the model.

## If you want to use your own dataset
Please refer to `guidance.docx`.

## Citation
If you use our code or dataset, please cite Raster-to-Graph:

Bibtex:
```
@article{https://doi.org/10.1111/cgf.15007,
author = {Hu, Sizhe and Wu, Wenming and Su, Ruolin and Hou, Wanni and Zheng, Liping and Xu, Benzhu},
year = {2024},
title = {Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer},
journal = {Computer Graphics Forum},
volume = {43},
number = {2},
pages = {e15007},
keywords = {CCS Concepts, • Computing methodologies → Shape modeling, Computer vision},
doi = {https://doi.org/10.1111/cgf.15007},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15007},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.15007}
}
```

We thank Raster-to-Vector for their contribution to our data. According to [Raster-to-Graph Terms of Use](https://drive.google.com/file/d/18n8aZeqSQ1nnQSfOnDg8dGr6yy2g00Oa0seqK0fs1fs/view), if you use our dataset, we suggest that you also cite Raster-to-Vector:

Bibtex:
```
@INPROCEEDINGS{8237503,
author={Liu, Chen and Wu, Jiajun and Kohli, Pushmeet and Furukawa, Yasutaka},
booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
title={Raster-to-Vector: Revisiting Floorplan Transformation},
year={2017},
volume={},
number={},
pages={2214-2222},
keywords={Junctions;Semantics;Computational modeling;Solid modeling;Linear
programming;Three-dimensional displays;IP networks},
doi={10.1109/ICCV.2017.241}
}
```

## Contact Me 
  If you have any questions, please contact me (Sizhe Hu) at [2024010072@mail.hfut.edu.cn](mailto:2024010072@mail.hfut.edu.cn). 
