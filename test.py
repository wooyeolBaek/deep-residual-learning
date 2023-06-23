import os
import json
import random
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchmetrics

from argparse import ArgumentParser
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomCIFAR10, get_test_dataset

import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = ArgumentParser()
    
    # python train.py --model_name resnet110 --normalization True --include_valid True --mapping A
    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--test_batch_size', type=int, default=128)

    # --dataset vars
    parser.add_argument('--per_pixel_mean_sub', type=bool, default=True)
    parser.add_argument('--per_pixel_std_div', type=bool, default=False)
    parser.add_argument('--whitening', type=bool, default=False)
    parser.add_argument('--normalization', type=bool, default=True)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--include_valid', type=bool, default=True)

    # --model vars
    parser.add_argument('--model_dir', type=str, default='./trained_models/resnet20_A_230622_192331')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet32_A_230622_192327')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet44_A_230622_192323')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet56_A_230622_192321')
    # parser.add_argument('--model_dir', type=str, default='./trained_models/resnet110_A_230622_192317')

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

def test(args, model, test_loader):
    # test loop
    model.eval()
    top1_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=1).to(args.device)
    top5_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=5).to(args.device)

    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs = inputs.type(torch.float32).to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)

            top1_metric.update(outputs, labels)
            top5_metric.update(outputs, labels)

        top1_acc = top1_metric.compute().item()
        top5_acc = top5_metric.compute().item()
        top1_acc *= 100
        top5_acc *= 100
        top1_metric.reset()
        top5_metric.reset()

        results_dict = {
            'top1_acc': round(top1_acc, 2),
            'top5_acc': round(top5_acc,2),
            'top1_error': round(100 - top1_acc, 2),
            'top5_error': round(100 - top5_acc, 2),
        }
        
        return results_dict


if __name__ == '__main__':
    args = parse_args()

    test_transform = A.Compose([
        ToTensorV2()
    ])

    test_X, test_Y = get_test_dataset(normalization=args.normalization)
    test_dataset = CustomCIFAR10(test_X, test_Y, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1
    )

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
    checkpoint = torch.load(os.path.join(args.model_dir, 'last.pth'))
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    result_dict = test(args, model, test_loader)

    with open(os.path.join(args.model_dir, 'test.json'), "w") as f:
        json.dump(result_dict, f, indent=2)

    print(result_dict)
