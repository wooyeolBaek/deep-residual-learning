#  Pretrained models

ResNet models pretrained with CIPHAR10


|   |  Paper's CIPHAR10<br> Top1-error[%] | Checkpoint's CIPHAR10<br> Top1-error[%] |
| :------------: | :--------------: | :--------------: |
ResNet-20| 8.75 | 9.18 |
ResNet-32| 7.51 | 8.68 |
ResNet-44| 7.17 | 8.13 |
ResNet-56| 6.97 | 7.10 |
ResNet-110| 6.43 | 6.95 |


### Settings

#### Train
- dataset: CIPHAR10(train=True 45k)
- pre-process: Per-pixel mean subtraction
- augmentation: paper's simple augmentation
    - 4 pixel pad
    - HorizontalFlip
    - 32x32 RandomCrop
- batch size: 128
- learning rate: 1e-1(divide by 10 at 34,000k and 48,000k, total = 64,000k)
- weight decay: 9e-1
- momentum: 1e-4
- etc: He initialization, Batch norm, no Dropout

#### Valid
- pre-process: Per-pixel mean subtraction
- dataset: CIPHAR10(train=True 5k)

#### Test
- pre-process: Per-pixel mean subtraction
- dataset: CIPHAR10(train=False 10k)