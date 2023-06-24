import os
import json
import random
from tqdm import tqdm

import torch
import numpy as np

from argparse import ArgumentParser
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = ArgumentParser()

    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2023)

    # --dataset vars
    parser.add_argument('--per_pixel_mean_sub', type=bool, default=True)
    parser.add_argument('--per_pixel_std_div', type=bool, default=False)
    parser.add_argument('--whitening', type=bool, default=False)
    parser.add_argument('--normalization', type=bool, default=True)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--include_valid', type=bool, default=True)

    # --model vars
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet20_A_230622_192331')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet32_A_230622_192327')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet44_A_230622_192323')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet56_A_230622_192321')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet110_A_230622_192317')

    parser.add_argument('--model_dir', type=str, default='./trained_models/plainnet20_A_230623_001026')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/plainnet32_A_230623_001020')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/plainnet44_A_230623_001008')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/plainnet56_A_230623_000836')

    # --etc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_feature_maps(model, image):
    image = image.unsqueeze(0)
    # image save
    # print('shape!!!',preprocessed_image.shape)

    # 특성 맵 저장을 위한 리스트
    feature_maps = []

    # 특성 맵 추출을 위한 hook 함수
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # 각 레이어에 hook 등록
    # shortcut.shortcut = ZeroPadMap
    # shortcut = IdentityMap
    layers = ('relu', 'shortcut', 'conv1')
    hooks = []
    hook_names = []
    for name, module in model.named_modules():
        if name.split('.')[-1] in layers:
            if name == 'module.conv1.relu':
                continue
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            hook_names.append(name)

    # 모델에 입력 이미지 전달하여 특성 맵 추출
    model.eval()
    with torch.no_grad():
        _ = model(image)

    # 등록한 hook 제거
    for hook in hooks:
        hook.remove()

    return feature_maps, hook_names


if __name__ == '__main__':
    args = parse_args()



    cfg = json.load(open(os.path.join(args.model_dir, 'config.json'), 'r'))
    model_name = cfg['model_name']
    mapping = cfg['mapping']
    block_name = cfg['block_name']
    model_module = getattr(import_module('models.resnet'), model_name)
    model = model_module(
        num_classes=10,
        mapping=mapping,
        block_name=block_name,
    )
    checkpoint = torch.load(os.path.join(args.model_dir, 'last.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)
    

    # makedirs feature map folder
    if 'resnet' in args.model_dir:
        dir = './resnet_feature_maps'
    else:
        dir = './plain_feature_maps'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    cifar10 = CIFAR10(root='data/', download=True, train=False)
    image, label = cifar10[1]
    image = np.array(image).astype(np.float32)
    plt.imsave(os.path.join(dir, 'input_image.png'), image.astype(np.uint8))
    preprocess = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 입력 이미지 전처리
    preprocessed_image = preprocess(image=image)['image']
    feature_maps, hook_names = get_feature_maps(model, preprocessed_image)

    # feature map 개수
    print(f"Total feature maps: {len(feature_maps)}")
    for i, (name, feature_map) in tqdm(enumerate(zip(hook_names, feature_maps)), total=len(feature_maps)):
        feature_map = feature_map.squeeze(0).cpu().numpy()

        nrow = 8 if feature_map.shape[0] == 64 else 4
        ncol = feature_map.shape[0] // nrow
        fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
        for j, feature in enumerate(feature_map):
            axes[j//ncol, j%ncol].imshow(feature, cmap='gray')
        

        if name.endswith('shortcut.shortcut'):
            name = '_'.join(name.split('.')[2:4]) + '_ZeroPadMap'
        elif name.endswith('shortcut'):
            name = '_'.join(name.split('.')[2:4]) + '_IdentityMap'
        elif name.endswith('relu'):
            name = '_'.join(name.split('.')[2:4]) + '_3x3ConvBnReLU'
        else:
            name = name.split('.')[1]
        
        # fig 제목 설정
        model_name = args.model_dir.split('/')[-1].split('_')[0]
        fig.suptitle(model_name +'_' + name, fontsize=16)

        plt.savefig(os.path.join(dir,f'{name}.png'))