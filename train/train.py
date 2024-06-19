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
from evaluation.eval import eval_f30k, eval_coco
from module.balancer import AdaptiveLossWeighting
from utils.builder import load_data, wandb_init, model_init

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
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
    l1_criterion = nn.L1Loss()
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

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

            # contrastive loss
            con_loss, con_img_loss, con_txt_loss = model.module.contrastive_loss(image_features, text_features
                                                                             , i_k, t_k)

            # l1 loss
            l1_img_loss = l1_criterion(image_features, tea_img_fea)
            l1_txt_loss = l1_criterion(text_features, tea_txt_fea)
            l1_loss = l1_img_loss + l1_txt_loss

            # cosine similarity loss
            cos_img_loss = 1 - ((cosine_similarity(image_features, tea_img_fea).mean()+1)/2)
            cos_txt_loss = 1 - ((cosine_similarity(text_features, tea_txt_fea).mean()+1)/2)
            cos_loss = cos_img_loss + cos_txt_loss

            # hard negative loss
            img_kd_tript_loss = model.module.tript_loss(image_features, tea_img_fea)
            txt_kd_tript_loss = model.module.tript_loss(text_features, tea_txt_fea)
            i2t_kd_tript_loss = model.module.tript_loss(image_features, tea_txt_fea)
            t2i_kd_tript_loss = model.module.tript_loss(text_features, tea_img_fea)
            tript_loss = img_kd_tript_loss + txt_kd_tript_loss + i2t_kd_tript_loss + t2i_kd_tript_loss
            

            current_losses = torch.tensor([l1_loss, cos_loss, con_loss, tript_loss])
            weights = adapter(current_losses).cuda()
            loss = l1_loss*weights[0] + cos_loss*weights[1] + con_loss*weights[2] + tript_loss*weights[3]
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module._dequeue_and_enqueue(i_k, t_k)

            if not args.debug:
                wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})
                

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
