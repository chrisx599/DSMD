import numpy as np
import torch
import time
import wandb
from tqdm import tqdm
import json

def eval_f30k(model, val_loader):
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
        text = torch.cat(cap_embs, dim=0)
        image = torch.cat(img_embs, dim=0)

    image_de = []
    for i in range(0, image.shape[0], 5):
        image_de.append(image[i])

    image = torch.stack(image_de, dim=0)



    start = time.time()

    scores = torch.matmul(image, text.T)
    end = time.time()
    print("calculate similarity time: {}".format(end - start))


    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    N = image.shape[0]

    for i in range(N):
        target_indices = list(range(i * 5, i * 5 + 5))
        
        _, indices = scores[i].topk(10, largest=True)
        indices = indices.cpu().numpy()
        
        recall_at_1 += np.intersect1d(target_indices, indices[:1]).size > 0
        recall_at_5 += np.intersect1d(target_indices, indices[:5]).size > 0
        recall_at_10 += np.intersect1d(target_indices, indices[:10]).size > 0

    average_recall_at_1 = recall_at_1 / N * 100
    average_recall_at_5 = recall_at_5 / N * 100
    average_recall_at_10 = recall_at_10 / N * 100

    average_recall_at_1, average_recall_at_5, average_recall_at_10

    recall_at_1_t2i = 0
    recall_at_5_t2i = 0
    recall_at_10_t2i = 0

    scores_t2i = scores.T

    for i in range(N * 5):
        target_index = i // 5

        _, indices = scores_t2i[i].topk(10, largest=True)
        indices = indices.cpu().numpy()

        recall_at_1_t2i += target_index in indices[:1]
        recall_at_5_t2i += target_index in indices[:5]
        recall_at_10_t2i += target_index in indices[:10]

    average_recall_at_1_t2i = recall_at_1_t2i / (N * 5) * 100
    average_recall_at_5_t2i = recall_at_5_t2i / (N * 5) * 100
    average_recall_at_10_t2i = recall_at_10_t2i / (N * 5) * 100

    average_recall_at_1_t2i, average_recall_at_5_t2i, average_recall_at_10_t2i
    


    print("Text to Image: %.1f, %.1f, %.1f" %
                 (average_recall_at_1_t2i, average_recall_at_5_t2i, average_recall_at_10_t2i))
    print("Image to Text: %.1f, %.1f, %.1f" %
                 (average_recall_at_1, average_recall_at_5, average_recall_at_10))
    # sum of recalls to be used for early stopping
    currscore = average_recall_at_1 + average_recall_at_5 + average_recall_at_10 + average_recall_at_1_t2i + average_recall_at_5_t2i + average_recall_at_10_t2i
    print('Current rsum is {}'.format(currscore))

    return currscore


def eval_coco(model, val_loader):
    img_embs = []
    cap_embs = []
    img_ids = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), desc='Progress'
                             , total=len(val_loader), unit='iter'):
            images = batch['image']
            texts = batch['text']
            ids = batch['image_id']
            images = images.to('cuda')
            texts = texts.to('cuda')
            image_features, text_features = model(images, texts.input_ids, texts.attention_mask)
            cap_embs.append(text_features)
            img_embs.append(image_features)
            img_ids.append(ids)
        text = torch.cat(cap_embs, dim=0)
        image = torch.cat(img_embs, dim=0)
        img_ids = torch.cat(img_ids, dim=0)

    image_feats = {}
    for ids, order in zip(img_ids, range(img_ids.shape[0])):
        idx = ids.item()
        if idx not in image_feats:
            image_feats[idx] = image[order]
    
    tiids = img_ids
    iids = []
    sorted_tensors = []
    for key in sorted(image_feats.keys()):
        sorted_tensors.append(image_feats[key].view(1, -1))
        iids.append(key)

    image_cls_feats = torch.cat(sorted_tensors, dim=0)

    scores = image_cls_feats @ text.t()
    iids = torch.LongTensor(iids).to(scores.device)
    tiids = tiids.to(scores.device)

    print("scores: {}".format(scores.size()))
    print("iids: {}".format(iids.size()))
    print("tiids: {}".format(tiids.size()))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0, 
        "tr_r5": tr_r5.item() * 100.0, 
        "tr_r1": tr_r1.item() * 100.0, 
        "ir_r10": ir_r10.item() * 100.0, 
        "ir_r5": ir_r5.item() * 100.0, 
        "ir_r1": ir_r1.item() * 100.0, 
        "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
    }

    print('* Eval result = %s' % json.dumps(eval_result))
    currscore = eval_result['average_score']

    return currscore
 
