
# Cross-Granularity Feature Fusion
 
Code release for A Cross-Granularity Feature Fusion Method for Fine-Grained Image Recogniton
 
### Requirement
 
python >= 3.7

PyTorch >= 1.3.1

torchvision >= 0.4.2

### Training

1. Download datatsets for FGVC (e.g. CUB-200-2011, Standford Cars, FGVC-Aircraft, etc) and organize the structure as follows:
```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```

2. Train from scratch with ``train.py``.

If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:

@article{wu2025cross,

         title={A cross-granularity feature fusion method for fine-grained image recognition},
  
         author={Wu, Shan and Hu, Jun and Sun, Chen and Zhong, Fujin and Zhang, Qinghua and Wang, Guoyin},
  
         journal={Applied Intelligence},
  
         volume={55},
  
         number={1},
  
         pages={1--19},
  
         year={2025},
  
         publisher={Springer}
  
}
