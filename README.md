![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# PVBLiF

Zhengyu Zhang, Shishun Tian, Wenbin Zou, Luce Morin, and Lu Zhang.

Official PyTorch code for our JSTSP2023 paper "PVBLiF: A pseudo video-based blind quality assessment metric for light field image". Please refer to our [paper](https://ieeexplore.ieee.org/abstract/document/10130290) for details.

**Note: We first convert the dataset into h5 files in MATLAB and then train/test the model in PYTHON.**

**Hope our work is helpful to you :)**

### Requirements
- PyTorch 1.7.1
- python 3.8

### Installation
Download this repository.
```
    $ git clone https://github.com/ZhengyuZhang96/PVBLiF.git
```

**(The following step takes the Win5-LID dataset for instance)**
### Generate Dataset in MATLAB 
After downloading the source dataset, use the PVBLiF_Win5_5x5_32x32.m in './PVBLiF/Datasets/...' to convert the dataset into h5 files, and then put them into './PVBLiF/Datasets/PVBLiF_Win5_5x5_32x32/...'.
```
    $ ./PVBLiF/Datasets/PVBLiF_Win5_5x5_32x32.m
```
or you can directly download the generated h5 files on [Baidu drive](https://pan.baidu.com/s/181NRPknx1_gKnsDBrnu8Sw) (code: INSA).

### Train
Train the model from scratch.
```
    $ python Train.py  --trainset_dir ./Datasets/PVBLiF_Win5_5x5_32x32/
```

### Test overall performance
Reproduce the performance in the paper: download our pre-trained models on [Baidu drive](https://pan.baidu.com/s/181NRPknx1_gKnsDBrnu8Sw) (code: INSA) and put them into './PVBLiF/PreTrainedModels/Win5/...'.
```
    $ python Test.py
```

### Test individual distortion type performance
Test the performance of individual distortion type using the following script. 
```
    $ python Test_Dist.py
```

### Results
The performances of our PVBLiF metric on the Win5-LID, NBU-LF1.0, and SHU datasets are provided as follows. Alternatively, you can reproduce these performances using the h5 results we provide in './PVBLiF/Results/...'.

**Win5-LID dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: |
|    HEVC      |  0.9768  |  0.9587  |  0.2215  |
|    JPEG2000  |  0.9388  |  0.9026  |  0.2887  |
|    LN        |  0.8941  |  0.8328  |  0.3324  |
|    NN        |  0.9286  |  0.8750  |  0.2526  |
|    Overall   |  0.8749  |  0.8580  |  0.4660  |

**NBU-LF1.0 dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: |
|    NN        |  0.9688  |  0.9162   |  0.1669  |
|    BI        |  0.9636  |  0.9304   |  0.2433  |
|    EPICNN    |  0.9498  |  0.8347   |  0.2146  |
|    Zhang     |  0.9011  |  0.8138   |  0.2350  |
|    VDSR      |  0.9605  |  0.9335   |  0.2386  |
|    Overall   |  0.9060  |  0.8883   |  0.3746  |

**SHU dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: |
|    GAUSS        |  0.9621  |  0.9524  |  0.2002  |
|    JPEG2000     |  0.9523  |  0.9477  |  0.1303  |
|    JPEG         |  0.9783  |  0.9619  |  0.2116  |
|    Motion Blur  |  0.9671  |  0.9574  |  0.2119  |
|    White Noise  |  0.9599  |  0.9580  |  0.2567  |
|    Overall      |  0.9554  |  0.9501  |  0.3160  |

### Citation
If you find this work helpful, please consider citing:
```
@article{PVBLiF,
  title        = {PVBLiF: A Pseudo Video-Based Blind Quality Assessment Metric for Light Field Image},
  author       = {Zhang, Zhengyu and Tian, Shishun and Zou, Wenbin and Morin, Luce and Zhang, Lu},
  journal      = {IEEE Journal of Selected Topics in Signal Processing},
  year         = {2023}
}
```

## Contact
Welcome to raise issues or email to [zhengyu.zhang@insa-rennes.fr](zhengyu.zhang@insa-rennes.fr) for any question regarding this work.
