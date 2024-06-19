import torch
import torch.nn as nn
import torch.optim as optim
from utils.argumants import get_args
import os
from tqdm import tqdm
import wandb
import time
import numpy as np
import os
from collections import OrderedDict
import torch.functional as F

from module.balancer import AdaptiveLossWeighting
from utils.builder import load_data, wandb_init, model_init, eval


def trainer():
    args = get_args()
    wandb_init(args)
    
    train_loader, val_loader, teacher_vectors_image, teacher_vectors_text = load_data(args)
    
    model = model_init(args)
    
    adapter = AdaptiveLossWeighting(4)

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

            tea_img_fea = teacher_vectors_image[i*args.batch_size:(i+1)*args.batch_size]
            tea_txt_fea = teacher_vectors_text[i*args.batch_size:(i+1)*args.batch_size]
            tea_img_fea = nn.functional.normalize(tea_img_fea, dim=1)
            tea_txt_fea = nn.functional.normalize(tea_txt_fea, dim=1)

            con_loss, l1_loss, cos_loss, hnd_loss = model.module.forward_loss(image_features, text_features, tea_img_fea, tea_txt_fea)

            current_losses = torch.tensor([l1_loss, cos_loss, con_loss, hnd_loss])
            weights = adapter(current_losses).cuda()
            loss = l1_loss*weights[0] + cos_loss*weights[1] + con_loss*weights[2] + hnd_loss*weights[3]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module._dequeue_and_enqueue(i_k, t_k)

            if args.wandb:
                wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})

            if (i + 1) % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        scheduler.step()
        torch.save(model.state_dict(), args.checkpoint_save_path)
        rsum = eval(model, args, val_loader)
        if rsum > best_rsum:
            torch.save(model.state_dict(), args.bestmodel_save_path)
            best_rsum = rsum
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}, Best R-sum: {best_rsum}')

    wandb.finish()


if __name__ == '__main__':
    trainer()
