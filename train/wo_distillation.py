import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from module.model import WO_Distillation_Model
from module.dataset import Flickr30kDataset, DataLoader, COCODataset
import argparse
from timm import create_model
import os
from tqdm import tqdm
import wandb
import time
import numpy as np
import os
from collections import OrderedDict
import torch.functional as F
from evaluation.eval import eval_f30k, eval_coco

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser('KDModel train', add_help=False)
    parser.add_argument('--task', type=str
                        , default='flickr30k', choices=['flickr30k', 'coco'])

    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64
                        , help='batch size')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--dataset', type=str, default='/root/autodl-tmp/dataset/flickr30k/')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--learn_rate', type=float, default=0.0001)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--learn_rate_update', type=int, default=7)
    parser.add_argument('--debug', type=bool, default=False 
                        , help='debug mode')
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoint_COCO.pth')
    parser.add_argument('--bestmodel_save_path', type=str, default='bestmodel_COCO.pth')
    parser.add_argument('--margin', default=0.05, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_violation', default=True,
                        help='Use max instead of sum in the rank loss.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    args.debug = True

    if not args.debug:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="kdmodel",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.learn_rate,
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "dim": args.dim,
            "seed": args.seed,
            "contrastive_t": 0.05,
            "checkpoint": args.checkpoint_save_path,
            "experiment": "self.T"
            }
        )

    # load data
    print("======>load data")

    if args.task == 'flickr30k':
        train_dataset = Flickr30kDataset(os.path.join(args.dataset, 'train.json')
                                        , os.path.join(args.dataset, 'flickr30k-images'))
        val_dataset = Flickr30kDataset(os.path.join(args.dataset, 'val.json')
                                    , os.path.join(args.dataset, 'flickr30k-images'))
    elif args.task == 'coco':
        train_dataset = COCODataset(os.path.join(args.dataset, 'train.json')
                                        , os.path.join(args.dataset, 'images'))
        val_dataset = COCODataset(os.path.join(args.dataset, 'val.json')
                                    , os.path.join(args.dataset, 'images'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size
                              , shuffle=False, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size
                            , shuffle=False, num_workers=args.num_workers)
    print("======>finish data prepare")

    # train
    model = KDModel(args)
    if args.resume:
        checkpoint = torch.load(args.resume)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print('=======>load checkpoint')
    model = torch.nn.DataParallel(model)
    model.module = model.module.cuda()
    

    decay_factor = 1e-4
    optimizer = optim.AdamW([
            {'params': model.module.txt_encoder.parameters(), 'lr': args.learn_rate},
            {'params': model.module.img_encoder.parameters(), 'lr': args.learn_rate}
        ],
            lr=args.learn_rate, weight_decay=decay_factor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learn_rate_update, gamma=0.1)


    best_rsum = 0
    for epoch in range(args.num_epochs):
        model.module.train()
        for i, batch in tqdm(enumerate(train_loader), desc='Progress'
                            , total=len(train_loader), unit='iter'):
            images = batch['image']
            texts = batch['text']
            images = images.to('cuda')
            texts = texts.to('cuda')
            image_features, text_features, i_k, t_k = model(images, texts.input_ids, texts.attention_mask)

            # contrastive loss
            loss, img_loss, txt_loss = model.module.contrastive_loss(image_features, text_features, i_k, t_k)


            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module._dequeue_and_enqueue(i_k, t_k)
            model.module._momentum_update_key_encoder()

            if not args.debug:
                wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]
                    , "epoch": epoch, "step": i
                    })
                

            if (i + 1) % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        scheduler.step()
        torch.save(model.state_dict(), args.checkpoint_save_path)
        model.module.eval()
        if args.task == 'coco':
            rsum = eval_coco(model, val_loader)
        elif args.task == 'flickr30k':
            rsum = eval_f30k(model, val_loader)
        if rsum > best_rsum:
            torch.save(model.state_dict(), args.bestmodel_save_path)
            best_rsum = rsum
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}, Best R-sum: {best_rsum}')

    wandb.finish()


if __name__ == '__main__':
    main()
