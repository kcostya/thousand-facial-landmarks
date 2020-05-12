### General Approach
* Warm-up with `OneCycleLR` policy and `SGD`
* Further training with `CosineAnnealingWarmRestarts` and `AdamW` with adaptive learning rates by layer
* Wing Loss
* ImageNet normalization
* Simple blend with another resnet50 model


### Basic Config
* CROP_SIZE = 128
* backbone = resnet50
* batch size = 32
* epochs = 5
* SGD_momentum = 0.9
* SGD_weight_decay = 1e-04
* OneCycle_max_lr = 0.1
* WingLoss_width = 10
* WingLoss_curvature = 2
* AdamW_weight_decay = 1e-06
* AdamW_amsgrad = True
* Adaptive learning rates:
```
[
    {"params": model.conv1.parameters(), "lr": 1e-6},
    {"params": model.bn1.parameters(), "lr": 1e-6},
    {"params": model.relu.parameters(), "lr": 1e-5},
    {"params": model.maxpool.parameters(), "lr": 1e-5},
    {"params": model.layer1.parameters(), "lr": 1e-4},
    {"params": model.layer2.parameters(), "lr": 1e-4},
    {"params": model.layer3.parameters(), "lr": 1e-3},
    {"params": model.layer4.parameters(), "lr": 1e-3},
    {"params": model.avgpool.parameters(), "lr": 1e-2},
    {"params": model.fc.parameters(), "lr": 1e-2},
]
```

### Run
```python train.py --name "baseline" --data "PATH_TO_DATA" [--gpu]```

Basic blending example is available within `blender.ipynb` [notebook](blender.ipynb)
