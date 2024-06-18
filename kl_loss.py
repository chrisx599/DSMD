import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from student_model import KDModel
from student_dataset import Flickr30kDataset, DataLoader
import argparse
from transformers import AlbertModel, AlbertTokenizer
from timm import create_model
import os
from torchvision import transforms
from tqdm import tqdm
import wandb
import time
import numpy as np
from evaluation import i2t, t2i, compute_sim
import os
from collections import OrderedDict
import torch.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser('KDModel train', add_help=False)
    parser.add_argument('--img_feats_path', type=str
                        , default='knowledgevec/f30k_beit3_base_image_feats.pkl')
    parser.add_argument('--txt_feats_path', type=str
                        , default='knowledgevec/f30k_beit3_base_text_feats.pkl')
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=128
                        , help='batch size')
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='/root/autodl-tmp/dataset/flickr30k')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--learn_rate', type=float, default=0.0001)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--learn_rate_update', type=int, default=20)
    parser.add_argument('--debug', type=bool, default=False
                        , help='debug mode')
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpointv4.pth')
    parser.add_argument('--bestmodel_save_path', type=str, default='bestmodel.pth')
    parser.add_argument('--con_lambda', type=float, default=1)
    parser.add_argument('--l1_lambda', type=float, default=1)
    parser.add_argument('--cos_lambda', type=float, default=1)
    parser.add_argument('--tript_lambda', type=float, default=1)
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')

    args = parser.parse_args()
    return args

def evaluate(model, val_loader):
    img_embs = []
    cap_embs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), desc='Progress'
                             , total=len(val_loader), unit='iter'):
            images = batch['image']
            texts = batch['text']
            images = images.to('cuda')
            texts = texts.to('cuda')
            image_features, text_features = model(images, texts.input_ids, texts.attention_mask)
            cap_embs.append(text_features)
            img_embs.append(image_features)
        cap_embs = torch.cat(cap_embs, dim=0)
        img_embs = torch.cat(img_embs, dim=0)
    cap_embs = cap_embs.cpu()
    img_embs = img_embs.cpu()
    cap_embs = cap_embs.numpy()
    img_embs = img_embs.numpy()
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])


    start = time.time()
    sims = compute_sim(img_embs, cap_embs)
    end = time.time()
    print("calculate similarity time: {}".format(end - start))

    # caption retrieval
    npts = img_embs.shape[0]
    # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    print('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    wandb.log({"r1": r1, "r5": r5, "r10": r10, 'medr': medr, 'meanr': meanr,
               "r1i": r1i, "r5i": r5i, "r10i": r10i, 'medr': medri, 'meanr': meanr, 
               'rsum':currscore})

    return currscore

def main():
    args = get_args()

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
            "con_lambda": args.con_lambda,
            "cos_lambda": args.cos_lambda,
            "l1_lambda": args.l1_lambda,
            "tript_lambda": args.tript_lambda
            }
        )

    # load data
    print("======>load data")
    with open(args.img_feats_path, 'rb') as f:
        teacher_vectors_image = pickle.load(f)
    with open(args.txt_feats_path, 'rb') as f:
        teacher_vectors_text = pickle.load(f)

    train_dataset = Flickr30kDataset(os.path.join(args.dataset, 'train.json')
                                     , os.path.join(args.dataset, 'flickr30k-images'))
    val_dataset = Flickr30kDataset(os.path.join(args.dataset, 'val.json')
                                   , os.path.join(args.dataset, 'flickr30k-images'))
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
    KLcriterion = nn.KLDivLoss(reduction='batchmean')
    decay_factor = 1e-4
    # optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
    optimizer = optim.AdamW([
            {'params': model.module.txt_encoder.parameters(), 'lr': args.learn_rate},
            {'params': model.module.img_encoder.parameters(), 'lr': args.learn_rate}
        ],
            lr=args.learn_rate, weight_decay=decay_factor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learn_rate_update, gamma=0.1)
    l1_criterion = nn.L1Loss()
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    # TODO: add cosine similarity to compute loss
    best_rsum = 0
    for epoch in range(args.num_epochs):
        model.module.train()
        for i, batch in tqdm(enumerate(train_loader), desc='Progress'
                            , total=len(train_loader), unit='iter'):
            images = batch['image']
            texts = batch['text']
            images = images.to('cuda')
            texts = texts.to('cuda')
            image_features, text_features = model(images, texts.input_ids, texts.attention_mask)

            tea_img_fea = teacher_vectors_image[i*args.batch_size:(i+1)*args.batch_size]
            tea_txt_fea = teacher_vectors_text[i*args.batch_size:(i+1)*args.batch_size]
            tea_img_fea = nn.functional.normalize(tea_img_fea, dim=1)
            tea_txt_fea = nn.functional.normalize(tea_txt_fea, dim=1)

            # contrastive loss
            con_loss, con_img_loss, con_txt_loss, con_i2t_loss, con_t2i_loss = model.module.contrastive_loss(image_features, text_features
                                                                             , tea_img_fea, tea_txt_fea)

            # l1 loss
            l1_img_loss = l1_criterion(image_features, tea_img_fea)
            l1_txt_loss = l1_criterion(text_features, tea_txt_fea)
            l1_i2t_loss = l1_criterion(image_features, tea_txt_fea)
            l1_t2i_loss = l1_criterion(text_features, tea_img_fea)
            l1_loss = l1_img_loss + l1_txt_loss + l1_i2t_loss + l1_t2i_loss

            # cosine similarity loss
            cos_img_loss = 1 - ((cosine_similarity(image_features, tea_img_fea).mean()+1)/2)
            cos_txt_loss = 1 - ((cosine_similarity(text_features, tea_txt_fea).mean()+1)/2)
            cos_i2t_loss = 1 - ((cosine_similarity(image_features, tea_txt_fea).mean()+1)/2)
            cos_t2i_loss = 1 - ((cosine_similarity(text_features, tea_img_fea).mean()+1)/2)
            cos_loss = cos_img_loss + cos_txt_loss + cos_i2t_loss + cos_t2i_loss

            # tript loss
            tript_loss = model.module.tript_loss(image_features, text_features)
            loss = args.con_lambda * con_loss + args.l1_lambda * l1_loss + args.cos_lambda * cos_loss + args.tript_lambda * tript_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module._dequeue_and_enqueue(tea_img_fea, tea_txt_fea)

            if not args.debug:
                wandb.log({"loss": loss, "con_loss": con_loss, "l1_loss": l1_loss, "cos_loss": cos_loss, "lr": scheduler.get_lr()[0]
                    , "l1_img_loss": l1_img_loss, "l1_txt_loss": l1_txt_loss, "l1_i2t_loss": l1_i2t_loss, "l1_t2i_loss": l1_t2i_loss
                    , "cos_img_loss": cos_img_loss, "cos_txt_loss": cos_txt_loss, "cos_i2t_loss": cos_i2t_loss, "cos_t2i_loss": cos_t2i_loss
                    , "con_img_loss": con_img_loss, "con_txt_loss": con_txt_loss, "con_i2t_loss": con_i2t_loss, "con_t2i_loss": con_t2i_loss
                    , "epoch": epoch, "step": i, "tript_loss": tript_loss
                    })

            if (i + 1) % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        scheduler.step()
        torch.save(model.state_dict(), args.checkpoint_save_path)
        model.module.eval()
        rsum = evaluate(model, val_loader)
        if rsum > best_rsum:
            torch.save(model.state_dict(), args.bestmodel_save_path)
            best_rsum = rsum
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}, Best R-sum: {best_rsum}')

        # update lambda
        if epoch == 3:
            args.tript_lambda = 0.001
            args.l1_lambda = 50
            args.cos_lambda = 50
        if epoch == 15:
            args.con_lambda = 20

    wandb.finish()


if __name__ == '__main__':
    main()
