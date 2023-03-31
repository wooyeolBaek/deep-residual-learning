import os
import math
import json
import random
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from sklearn.metrics import f1_score, accuracy_score

import wandb
from pytz import timezone
from datetime import datetime
from argparse import ArgumentParser
from importlib import import_module
from PIL import Image
from torchvision.transforms.functional import to_pil_image



import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--epochs', type=int, default=164)
    parser.add_argument('--max_iter', type=int, default=64000)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='SGD')

    # --experiment vars
    parser.add_argument('--mapping', type=str, default='A')
    parser.add_argument('--block_name', type=str, default='ResBlock')

    # --model vars
    parser.add_argument('--model', type=str, default='resnet', help='resnet, plain, vgg')
    parser.add_argument('--model_name', type=str, default='resnet18', help='resnet18, plain18, ...')

    # --etc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saved_dir", type=str, default="./trained_models")

    # --wandb configs
    parser.add_argument("--wandb_project", type=str, default="image_classification")
    parser.add_argument("--wandb_entity", type=str, default="wooyeolbaek")
    parser.add_argument("--wandb_run", type=str, default="exp")

    args = parser.parse_args()

    args.epochs = (args.max_iter * args.train_batch_size) // 50_000 + 1

    args.wandb_run = args.model_name + '_' + args.mapping

    return args

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class CustomNormalize(transforms.Normalize):
    def __init__(self, mean):
        super().__init__(mean=mean, std=[1, 1, 1])
        self.train_mean = mean

    def __call__(self, image):
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(self.train_mean,(1,1,1))(image)
        
        image = to_pil_image(image)

        return image

class PerPixelSubtraction:
    """per-pixel mean subtraction 기능
    """
    def __init__(self, pixel_mean):
        self.pixel_mean = pixel_mean

    
    def __call__(self, image):
        image = np.array(image)
        print('image shape',image.shape)
        print('pixel_mean shape',self.pixel_mean.shape)
        assert image.shape != self.pixel_mean.shape, f'image.shape:{image.shape}이 pixel_mean.shape:{self.pixel_mean.shape}와 일치하지 않습니다.'
        image = image - self.pixel_mean
        return Image.fromarray(image)
    
def get_pixel_mean():
    dataset = CIFAR10(
        root="./data/CIFAR10/",
        train=True,
        transform=None,
        download=True
    )
    cnt = 0
    result = np.zeros_like(dataset[0][0])
    for data in tqdm(dataset):
        image = np.array(data[0])
        result = result * (cnt/(cnt+1)) + image*(1/(cnt+1))
        cnt += 1

    return np.array(result).astype(np.uint8)

def train(args):
    seed_everything(args.seed)

    start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")
    saved_dir = os.path.join(args.saved_dir, args.wandb_run + start_time)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # config 설정
    with open(os.path.join(saved_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # wandb 설정
    wandb.init(
        project=f"{args.wandb_project}",
        entity=f"{args.wandb_entity}",
        name=args.wandb_run + start_time,
    )
    wandb.config.update(
        {
            "run_name": args.wandb_run,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "valid_batch_size": args.valid_batch_size,
            "criterion": args.criterion,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "seed": args.seed,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        }
    )

    train_dataset = CIFAR10(
        root="./data/CIFAR10/",
        train=True,
        download=True
    )
    valid_dataset = CIFAR10(
        root="./data/CIFAR10/",
        train=False,
        download=True
    )

    #pixel_mean = get_pixel_mean()

    train_mean = train_dataset.data.mean(axis=(0,1,2)) / 255
    train_std = train_dataset.data.std(axis=(0,1,2)) / 255

    train_transform = transforms.Compose([
        #PerPixelSubtraction(pixel_mean),
        #CustomNormalize(mean=train_mean),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(
            size=(32,32),
            padding=4,
            pad_if_needed=True,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=0,
            hue=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(train_mean,train_std),
        #transforms.Normalize(train_mean,(1,1,1)),
    ])
    valid_transform = transforms.Compose([
        #PerPixelSubtraction(pixel_mean),
        #CustomNormalize(mean=train_mean),
        transforms.ToTensor(),
        transforms.Normalize(train_mean,train_std),
        #transforms.Normalize(train_mean,(1,1,1)),
    ])

    train_dataset.transform = train_transform
    valid_dataset.transform = valid_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=True,
        num_workers=1
    )

    model_module = getattr(import_module(args.model), args.model_name)
    model = model_module(
        num_classes=10,
        mapping=args.mapping,
        block_name=args.block_name,
    )
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    #criterion = nn.CrossEntropyLoss
    criterion = getattr(import_module("torch.nn"), args.criterion)()

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    # --AMP
    scaler = torch.cuda.amp.GradScaler()

    iter = 0

    best_avg_valid_loss = np.inf
    best_valid_f1 = 0
    targets = []
    preds_list = []

    num_train_batches = math.ceil(len(train_dataset) / args.train_batch_size)

    with open(os.path.join(saved_dir,"log.txt"),"w") as f:
        for epoch in range(args.epochs):
            # train loop
            model.train()
            train_loss = 0
            train_f1 = 0

            with tqdm(total=num_train_batches) as pbar:
                for idx, (inputs, labels) in enumerate(train_loader):
                    iter += 1
                    pbar.set_description(f"[Train] Epoch [{epoch+1}/{args.epochs}]")

                    inputs = inputs.to(args.device)
                    
                    labels = labels.to(args.device)
                    

                    optimizer.zero_grad()

                    model = model.to(args.device)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=-1)
                    
                    
                    scaler.scale(loss).backward()
                    # gradient clipping
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    scaler.step(optimizer)
                    scaler.update()

                    targets += labels.tolist() # labels
                    preds_list += preds.tolist() # preds
                    train_f1 = f1_score(targets, preds_list, average='macro')
                    train_acc = accuracy_score(targets, preds_list)

                    pbar.update(1)

                    pbar.set_postfix(
                        {
                            "Loss": round(loss.item(), 4),
                            "Accuracy": round(train_acc, 4),
                            "Macro-F1": round(train_f1, 4),
                        }
                    )
                    
                    train_loss += loss.item()

                    if iter == args.max_iter:
                        break
                    if iter == 32000 or iter == 48000:
                        optimizer.param_groups[0]['lr'] /= 10
            
            train_log = "[EPOCH TRAIN {}/{}] : Train Loss {} - Train Accuracy {} - Train macro-f1 {}".format(
                epoch + 1,
                args.epochs,
                round(train_loss / len(train_loader), 4),
                round(train_acc, 4),
                round(train_f1, 4),
            )
            f.write(train_log + "\n")
            
            wandb.log(
                {
                    "train/loss": round(train_loss / len(train_loader), 4),
                    "train/accuracy": round(train_acc, 4),
                    "train/macro-f1": round(train_f1, 4),
                    "train/learning_rate": get_learning_rate(optimizer)
                }
            )

            # val loop
            with torch.no_grad():
                model.eval()

                valid_loss = 0
                valid_f1 = 0
                targets = []
                preds_list = []

                num_valid_batches = math.ceil(len(valid_dataset) / args.valid_batch_size)

                with tqdm(total=num_valid_batches) as pbar:
                    for step, (inputs, labels) in enumerate(valid_loader):
                        pbar.set_description(f"[Valid] Epoch [{epoch+1}/{args.epochs}]")

                        inputs = inputs.to(args.device)
                        
                        labels = labels.to(args.device)
                        

                        model = model.to(args.device)

                        outputs = model(inputs)
                        
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=-1)

                        targets += labels.tolist() # labels
                        preds_list += preds.tolist() # preds
                        valid_f1 = f1_score(targets, preds_list, average='macro')
                        valid_acc = accuracy_score(targets, preds_list)

                        pbar.update(1)

                        pbar.set_postfix(
                            {
                                "Loss": round(loss.item(), 4),
                                "Accuracy": round(valid_acc, 4),
                                "Macro-F1": round(valid_f1, 4),
                            }
                        )
                        
                        valid_loss += loss.item()

                avg_valid_loss = valid_loss / len(valid_loader)
                best_avg_valid_loss = min(best_avg_valid_loss, avg_valid_loss)

                valid_log = "[EPOCH VALID {}/{}] : VALID Loss {} - VALID Accuracy {} - VALID macro-f1 {}".format(
                    epoch + 1,
                    args.epochs,
                    round(valid_loss / len(valid_loader), 4),
                    round(valid_acc, 4),
                    round(valid_f1, 4),
                )
                f.write(valid_log + "\n")

                wandb.log(
                    {
                        "valid/loss": round(valid_loss / len(valid_loader), 4),
                        "valid/accuracy": round(valid_acc, 4),
                        "valid/macro-f1": round(valid_f1, 4),
                    }
                )

                
                if valid_f1 >= best_valid_f1:
                    print(f"New best model for val macro-f1 : {valid_f1:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{saved_dir}/best.pth")
                    best_valid_f1 = valid_f1
                    counter = 0
                else:
                    counter += 1
                
                torch.save(model.module.state_dict(), f"{saved_dir}/lastest.pth")
                print()

                


if __name__ == '__main__':
    args = parse_args()

    train(args)