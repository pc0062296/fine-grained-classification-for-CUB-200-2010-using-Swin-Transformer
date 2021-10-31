# fine-grained-classification-for-CUB-200-2010-using-Swin-Transformer
homework from NCTU VRDL class 2021 fall
By Ya-Shu Yang


## Dependencies:
+ Python 3.7.3
+ PyTorch 1.9.0
+ torchvision 0.10.0
+ timm 0.4.12

## Usage
### 1. Download pre-trained models from git



### 2. Prepare data

In the work, I use data from a publicly available datasets:

+ [CUB-200-2010](http://www.vision.caltech.edu/visipedia/CUB-200.html)

Since this code is for NCTU VRDL homework, the dataset is adjusted here.
Please download the training images and testing images from CodaLab competition:
https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836
Put the training images to dataset/train, and put the testing images to dataset/test



### 3. Install required packages

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

### 4. Train

To train on the dataset with 1 gpu in FP-16 mode for 10000 steps run:

```bash
python train.py
```

### 5. Reproduce

To Reproduce the result:

Please download the pre-trained model from:

http:
and put it output/hw1model

```bash
python inference.py
```


## Reference


```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jieneng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
