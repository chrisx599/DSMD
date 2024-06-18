"""动量对比学习编码器"""
import os
import torch.nn.init
from torch.nn.utils import clip_grad_norm_
from torch.nn import Parameter
import random
import logging
from collections import OrderedDict
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import pickle
from scipy import linalg
import torch.nn as nn

logger = logging.getLogger(__name__)

class Contrastive(nn.Module):
    def __init__(self, opt, dim=1024, K=4096, m=0.99, T=0.07):
        super(Contrastive, self).__init__()
        self.opt = opt
        self.K = K
        self.m = m
        self.T = T
        # 创建infoNCE loss中所需要的负样例队列
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("i_queue", torch.randn(dim, K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

        # encoders, 初始化全局嵌入阶段编码器
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.img_k_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        self.txt_k_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        # self.fc1_enc = nn.Linear(opt.embed_size, dim)
        # self.fc2_enc = nn.Linear(opt.embed_size, dim)
        # nn.init.xavier_normal_(self.fc1_enc.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc2_enc.weight, gain=1.414)

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        all_text_params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        text_params_no_bert = list()
        for p in all_text_params:
            if p.data_ptr() not in bert_params_ptr:
                text_params_no_bert.append(p)

        self.optimizer = torch.optim.AdamW([
            {'params': text_params_no_bert, 'lr': opt.learning_rate},
            {'params': bert_params, 'lr': opt.learning_rate*0.1},
            {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
        ],
            lr=opt.learning_rate, weight_decay=decay_factor)
        

        for param_q, param_k in zip(self.img_enc.parameters(), self.img_k_enc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.txt_enc.parameters(), self.txt_k_enc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    def forward(self, images, image_lengths, captions, lengths):
        # 全局嵌入阶段
        img_emb, _ = self.img_enc(images, image_lengths) # B*dim_emb, B*max_num_img_features*dim_emb
        cap_emb, _ = self.txt_enc(captions, lengths)  # B*dim_emb, B*L*dim_emb, L is the max length of caps in mini-batch

        img_bit_emb=l2norm(img_emb, dim=-1)
        cap_bit_emb=l2norm(cap_emb, dim=-1)

        with torch.no_grad():
            img_k_emb, _ = self.img_k_enc(images, image_lengths)
            cap_k_emb, _ = self.txt_k_enc(captions, lengths)
            img_bit_emb_k=l2norm(img_k_emb, dim=-1)
            cap_bit_emb_k=l2norm(cap_k_emb, dim=-1)


        return img_bit_emb, cap_bit_emb, img_bit_emb_k, cap_bit_emb_k


class ContrastiveModel(object):
    def __init__(self, opt):
        # 创建模型
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.feature_fuse_type = opt.feature_fuse_type

        self.model=Contrastive(opt)

        if torch.cuda.is_available():
            self.model.cuda()
            cudnn.benchmark = True

        self.params = self.model.parameters()

        self.Eiters = 0
        self.model = nn.DataParallel(self.model)
        self.optimizer = self.model.module.optimizer
        logger.info('The model is data paralleled now.')

    def state_dict(self):
        return self.model.module.state_dict()

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)

    def train_start(self):
        self.model.train()

    def val_start(self):
        self.model.eval()

    def forward_emb(self, images, image_lengths, captions, lengths):

        """
        Compute the image and caption embeddings
        fdd: add params: concept_labels, concept_input_embs, alpha,
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()  # B*C*D, C<=36, D is the dimension BUTD feature(default 2048)
            captions = captions.int().cuda()  # B*L, int, L is the max length of caps in mini-batch
            image_lengths = image_lengths.cuda()  # B, int
            lengths = torch.Tensor(lengths).cuda()  # B, int
        if self.opt.precomp_enc_type == 'basic':
            emb_v, emb_t, emb_v_k, emb_t_k \
            = self.model(images, image_lengths, captions, lengths)
        else:
            raise ValueError("opt.precomp_enc_type must be 'basic' currently, it is {} now".format(self.opt.precomp_enc_type))

        return emb_v, emb_t, emb_v_k, emb_t_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.module.img_enc.parameters(), 
                                    self.model.module.img_k_enc.parameters()):
            param_k.data = param_k.data * self.model.module.m + param_q.data * (1. - self.model.module.m)

        for param_q, param_k in zip(self.model.module.txt_enc.parameters(), 
                                    self.model.module.txt_k_enc.parameters()):
            param_k.data = param_k.data * self.model.module.m + param_q.data * (1. - self.model.module.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, i_keys, t_keys):
        batch_size = i_keys.shape[0]
        i_ptr = int(self.model.module.i_queue_ptr)
        t_ptr = int(self.model.module.t_queue_ptr)
        assert self.model.module.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.model.module.i_queue[:, i_ptr:i_ptr + batch_size] = i_keys.t()  # transpose
        i_ptr = (i_ptr + batch_size) % self.model.module.K  # move pointer

        self.model.module.t_queue[:, t_ptr:t_ptr + batch_size] = t_keys.t()  # transpose
        t_ptr = (t_ptr + batch_size) % self.model.module.K  # move pointer

        self.model.module.i_queue_ptr[0] = i_ptr
        self.model.module.t_queue_ptr[0] = t_ptr


    def forward_loss(self, i_q, t_q, i_k, t_k):
        """Compute the loss given pairs of image and caption embeddings
        """

        # # N*dim, N is batch size
        # ii_pos = torch.einsum('nc,nc->n', [i_q, i_aug_k]).unsqueeze(-1)
        # tt_pos = torch.einsum('nc,nc->n', [t_q, t_aug_k]).unsqueeze(-1)
        # ii_neg = torch.einsum('nc,ck->nk', [i_q, self.model.module.i_queue.clone().detach()])
        # tt_neg = torch.einsum('nc,ck->nk', [t_q, self.model.module.t_queue.clone().detach()])
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # negative logits: NxK
        i2t_pos = torch.einsum('nc,nc->n', [i_q, t_k]).unsqueeze(-1)
        i2t_neg = torch.einsum('nc,ck->nk', [i_q, self.model.module.t_queue.clone().detach()])
        t2i_pos = torch.einsum('nc,nc->n', [t_q, i_k]).unsqueeze(-1)
        t2i_neg = torch.einsum('nc,ck->nk', [t_q, self.model.module.i_queue.clone().detach()])

        # logits: Nx(1+K)
        i2t_logits = torch.cat([i2t_pos, i2t_neg], dim=1)
        t2i_logits = torch.cat([t2i_pos, t2i_neg], dim=1)
        i2t_logits = i2t_logits / self.model.module.T
        t2i_logits = t2i_logits / self.model.module.T
        labels = torch.zeros(t2i_logits.shape[0], dtype=torch.long).cuda()
        i2t_loss = nn.CrossEntropyLoss().cuda()(i2t_logits, labels)
        t2i_loss = nn.CrossEntropyLoss().cuda()(t2i_logits, labels)

        loss = i2t_loss + t2i_loss
        # intra_loss = tt_loss + ii_loss
        # a = 0.9
        # loss = a*inter_loss + (1-a)*intra_loss
        

        self.logger.update('Loss', loss.item(), i_q.size(0))
        return loss

    def train_emb(self, images, img_lengths, captions, lengths):
        """One training step given images and captions.
        """

        self.Eiters += 1
        self.logger.update(' Eit', self.Eiters)
        self.logger.update(' lr', self.optimizer.param_groups[0]['lr'])

        # 计算全局嵌入
        i_q, t_q, i_k, t_k = \
            self.forward_emb(images, img_lengths, captions, lengths)

        self.optimizer.zero_grad()
        loss = self.forward_loss(i_q, t_q, i_k, t_k)
        loss.backward()
        self.optimizer.step()
        self._momentum_update_key_encoder()
        self._dequeue_and_enqueue(i_k, t_k)