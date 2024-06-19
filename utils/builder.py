import pickle
from module.dataset import Flickr30kDataset, COCODataset
from torch.utils.data import DataLoader
import os
import wandb
import torch
from collections import OrderedDict
from module.model import KDModel

def load_data(args):
    print("===============>load data")
    with open(args.img_feats_path, 'rb') as f:
        teacher_vectors_image = pickle.load(f)
    with open(args.txt_feats_path, 'rb') as f:
        teacher_vectors_text = pickle.load(f)

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
    print("===============>finish data prepare")

    return train_loader, val_loader, teacher_vectors_image, teacher_vectors_text

def wandb_init(args):
    if args.wandb:
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
            }
        )

def model_init(args):
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
    return model
