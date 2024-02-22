
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


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
   hujun@cqupt.edu.cn
   wushanj2023@163.com
