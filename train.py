import os
import math
import json
import random
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

import wandb
from pytz import timezone
from datetime import datetime
from argparse import ArgumentParser
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomCIFAR10, preprocess

import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epochs', type=int, default=164)
    parser.add_argument('--max_iter', type=int, default=64000)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=9e-1)
    parser.add_argument('--optimizer', type=str, default='SGD')

    # --experiment vars
    parser.add_argument('--mapping', type=str, default='A')
    parser.add_argument('--block_name', type=str, default='ResBlock')

    # --dataset vars
    parser.add_argument('--whitening', type=bool, default=False)
    parser.add_argument('--normalization', type=bool, default=False)
    parser.add_argument('--epsilon', type=float, default=0.1)

    # --model vars
    parser.add_argument('--model', type=str, default='resnet', help='resnet, plain, vgg')
    parser.add_argument('--model_name', type=str, default='resnet18', help='resnet18, plain18, ...')

    # --etc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saved_dir", type=str, default="./trained_models")
    parser.add_argument("--amp", type=bool, default=True)

    # --wandb configs
    parser.add_argument("--wandb_project", type=str, default="image_classification")
    parser.add_argument("--wandb_entity", type=str, default="wooyeolbaek")
    parser.add_argument("--wandb_run", type=str, default="exp")

    args = parser.parse_args()

    args.epochs = (args.max_iter * args.train_batch_size) // 45_000 + 1

    args.wandb_run = args.model_name + '_' + args.mapping

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

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_topk_scores(outputs, labels, k):
    assert len(labels)==len(outputs), f'len(labels)={len(labels)} should be the same as len(outputs)={len(outputs)}'
    topk_scores = []
    for output, label in zip(outputs, labels):
        topk_score = torch.topk(torch.tensor(output), k)

        topk_scores.append(1 if label.item() in [v.item() for v in topk_score[1]] else 0)
    
    return topk_scores

def train(args, model, criterion, optimizer, train_loader, valid_loader):
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

    model.to(args.device)
    criterion.to(args.device)

    # --AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_iteration = 0
    best_mean_valid_loss = np.inf

    with open(os.path.join(saved_dir,"log.txt"),"w") as f:
        for epoch in range(args.epochs):

            # --train
            model.train()
            train_loss = []
            top1_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=1).to(args.device)
            top5_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=5).to(args.device)

            num_train_batches = math.ceil(len(train_dataset) / args.train_batch_size)
            with tqdm(total=num_train_batches) as pbar:
                for step, (inputs, labels) in enumerate(train_loader):
                    pbar.set_description(f"[Train] Epoch [{epoch+1}/{args.epochs}]")

                    train_iteration += 1
                    inputs1 = inputs[:args.train_batch_size//2].type(torch.float32).to(args.device)
                    inputs2 = inputs[args.train_batch_size//2:].type(torch.float32).to(args.device)
                    labels = labels.to(args.device)

                    with torch.cuda.amp.autocast():
                        outputs1 = model(inputs1)
                        outputs2 = model(inputs2)
                        outputs = torch.cat([outputs1, outputs2], dim=0)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    top1_metric.update(outputs, labels)
                    top5_metric.update(outputs, labels)
                    mean_loss = loss.item() / len(labels)
                    train_loss.append(loss.item())

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": round(mean_loss, 4),
                            "iter": train_iteration,
                        }
                    )
                    
                    if train_iteration == args.max_iter:
                        break
                    if train_iteration==32000 or train_iteration==48000:
                        optimizer.param_groups[0]['lr'] /= 10

            mean_train_loss = np.mean(train_loss)
            
            top1_acc = top1_metric.compute().item()
            top5_acc = top5_metric.compute().item()
            top1_error = 100*(1-top1_acc)
            top5_error = 100*(1-top5_acc)
            top1_metric.reset()
            top5_metric.reset()

            train_log = "[EPOCH TRAIN {}/{}] : Train loss {} - Train top1_acc {} - Train top5_acc {}".format(
                epoch + 1,
                args.epochs,
                round(mean_train_loss, 4),
                round(top1_acc, 4),
                round(top5_acc, 4),
            )
            f.write(train_log + "\n")

            results_dict = validation(args, epoch, model, criterion, valid_loader)
            f.write(results_dict["valid_log"] + "\n")
            
            if results_dict["mean_valid_loss"] <= best_mean_valid_loss:
                print(f'New best model : {results_dict["mean_valid_loss"]:4.2%}! saving the best model..')
                torch.save(model.module.state_dict(), f"{saved_dir}/best_loss.pth")
                best_mean_valid_loss = results_dict["mean_valid_loss"]
                counter = 0
            else:
                counter += 1
            
            torch.save(model.module.state_dict(), f"{saved_dir}/lastest.pth")
            print()

            wandb.log(
                {
                    "train/loss": round(mean_train_loss, 4),
                    "train/top1_error": top1_error,
                    "train/top5_error": top5_error,
                    "train/top1_acc": round(top1_acc, 4),
                    "train/top5_acc": round(top5_acc, 4),

                    "valid/loss": round(results_dict["mean_valid_loss"], 4),
                    "valid/top1_error": results_dict["top1_error"],
                    "valid/top5_error": results_dict["top5_error"],
                    "valid/top1_acc": round(results_dict["top1_acc"], 4),
                    "valid/top5_acc": round(results_dict["top5_acc"], 4),

                    "iter": train_iteration,
                    "learning_rate": get_learning_rate(optimizer),
                }
            )

def validation(args, epoch, model, criterion, valid_loader):
    # val loop
    model.eval()
    valid_loss = []
    top1_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=1).to(args.device)
    top5_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=5).to(args.device)

    with torch.no_grad():
        num_valid_batches = math.ceil(len(valid_dataset) / args.valid_batch_size)
        with tqdm(total=num_valid_batches) as pbar:
            for step, (inputs, labels) in enumerate(valid_loader):
                pbar.set_description(f"[Valid] Epoch [{epoch+1}/{args.epochs}]")

                inputs = inputs.type(torch.float32).to(args.device)
                labels = labels.to(args.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                top1_metric.update(outputs, labels)
                top5_metric.update(outputs, labels)
                mean_loss = loss.item() / len(labels)
                valid_loss.append(loss.item())

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": round(mean_loss, 4),
                    }
                )
                
        mean_valid_loss = np.mean(valid_loss)

        top1_acc = top1_metric.compute().item()
        top5_acc = top5_metric.compute().item()
        top1_error = 100*(1-top1_acc)
        top5_error = 100*(1-top5_acc)
        top1_metric.reset()
        top5_metric.reset()
        
        valid_log = "[EPOCH Valid {}/{}] : Valid loss {} - Valid top1_acc - Valid top5_acc {}".format(
            epoch + 1,
            args.epochs,
            round(mean_valid_loss, 4),
            round(top1_acc, 4),
            round(top5_acc, 4),
        )
        results_dict = {
            "top1_error": top1_error,
            "top5_error": top5_error,
            "mean_valid_loss": mean_valid_loss,
            "top1_acc": round(top1_acc, 4),
            "top5_acc": round(top5_acc, 4),
            "valid_log": valid_log
        }
        return results_dict


if __name__ == '__main__':
    args = parse_args()

    train_transform = A.Compose([
        A.PadIfNeeded(
            min_height=40,
            min_width=40,
            value=0,
            p=1.0
        ),
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5),
        # A.FancyPCA(
        #     alpha=0.1,
        #     always_apply=False,
        #     p=0.5
        # ),
        ToTensorV2()
    ])
    valid_transform = A.Compose([
        ToTensorV2()
    ])

    train_X, valid_X, train_Y, valid_Y = preprocess(
        whitening=args.whitening,
        normalization=args.normalization,
        epsilon=args.epsilon
    )
    train_dataset = CustomCIFAR10(train_X, train_Y, transform=train_transform)
    valid_dataset = CustomCIFAR10(valid_X, valid_Y, transform=valid_transform)
    
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
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad()

    train(args, model, criterion, optimizer, train_loader, valid_loader)