import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from transformers import BertModel, BertTokenizer
from module.loss import TriptLoss, ContrastiveLoss
    
class Bert_base(nn.Module):
    def __init__(self, output_dim):
        super(Bert_base, self).__init__()
        model_name = "google/bert_uncased_L-12_H-768_A-12"
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
    


class ViT(nn.Module):
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=config)
        self.fc = nn.Linear(config.hidden_size, output_dim)
        
    def forward(self, image):
        outputs = self.vit(image)
        cls_output = outputs.last_hidden_state[:, 0, :]
        features = self.fc(cls_output)
        
        return features



class KDModel(nn.Module):
    def __init__(self, args):
        super(KDModel, self).__init__()
        output_dim = args.dim
        self.img_encoder = ViT(output_dim=output_dim)
        self.txt_encoder = Bert_base(output_dim=output_dim)
        nn.init.xavier_normal_(self.img_encoder.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.txt_encoder.fc.weight, gain=1.414)

        # register buffer for queue of constrastive learning
        self.T = 0.05
        self.dim = output_dim
        self.K = 8192
        self.register_buffer("t_queue", torch.randn(self.dim, self.K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("i_queue", torch.randn(self.dim, self.K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

        # define tript loss
        self.con = ContrastiveLoss(t=self.T)
        self.hnd = TriptLoss(opt=args)
        self.l1 = nn.L1Loss()
        self.cos= nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, images, input_ids, attention_mask):
        img_fea = self.img_encoder(images)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        txt_fea = self.txt_encoder(input_ids, attention_mask)
        img_fea = nn.functional.normalize(img_fea, dim=1)
        txt_fea = nn.functional.normalize(txt_fea, dim=1)
        return img_fea, txt_fea
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, i_keys, t_keys):
        batch_size = i_keys.shape[0]
        i_ptr = int(self.i_queue_ptr)
        t_ptr = int(self.t_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.i_queue[:, i_ptr:i_ptr + batch_size] = i_keys.t()  # transpose
        i_ptr = (i_ptr + batch_size) % self.K  # move pointer

        self.t_queue[:, t_ptr:t_ptr + batch_size] = t_keys.t()  # transpose
        t_ptr = (t_ptr + batch_size) % self.K  # move pointer

        self.i_queue_ptr[0] = i_ptr
        self.t_queue_ptr[0] = t_ptr

    def forward_loss(self, stu_img, stu_txt, tea_img, tea_txt):
        # contrastive loss
        con_loss = self.contrastive_loss(stu_img, stu_txt, tea_img, tea_txt)

        # l1 loss
        l1_img_loss = self.l1(stu_img, tea_img)
        l1_txt_loss = self.l1(stu_txt, tea_txt)
        l1_loss = l1_img_loss + l1_txt_loss

        # cosine similarity loss
        cos_img_loss = 1 - ((self.cos(stu_img, tea_img).mean()+1)/2)
        cos_txt_loss = 1 - ((self.cos(stu_txt, tea_txt).mean()+1)/2)
        cos_loss = cos_img_loss + cos_txt_loss

        # hard negative loss
        img_kd_tript_loss = self.tript_loss(stu_img, tea_img)
        txt_kd_tript_loss = self.tript_loss(stu_txt, tea_txt)
        i2t_kd_tript_loss = self.tript_loss(stu_img, tea_txt)
        t2i_kd_tript_loss = self.tript_loss(stu_txt, tea_img)
        tript_loss = img_kd_tript_loss + txt_kd_tript_loss + i2t_kd_tript_loss + t2i_kd_tript_loss

        return con_loss, l1_loss, cos_loss, tript_loss


class WO_Distillation_Model(nn.Module):
    def __init__(self, args):
        super(WO_Distillation_Model, self).__init__()
        output_dim = args.dim
        self.img_encoder = ViT(output_dim=output_dim)
        self.txt_encoder = Bert_base(output_dim=output_dim)
        self.img_encoder_k = ViT(output_dim=output_dim)
        self.txt_encoder_k = Bert_base(output_dim=output_dim)

        for param_q, param_k in zip(self.img_encoder.parameters(), self.img_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.txt_encoder.parameters(), self.txt_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # register buffer for queue of constrastive learning
        self.T = 0.05
        self.dim = output_dim
        self.K = 8192
        self.register_buffer("t_queue", torch.randn(self.dim, self.K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("i_queue", torch.randn(self.dim, self.K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss = TriptLoss(args)
        self.m = 0.99

    def forward(self, images, input_ids, attention_mask):
        img_fea = self.img_encoder(images)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        txt_fea = self.txt_encoder(input_ids, attention_mask)
        img_fea = nn.functional.normalize(img_fea, dim=1)
        txt_fea = nn.functional.normalize(txt_fea, dim=1)

        img_fea_k = self.img_encoder_k(images)
        txt_fea_k = self.txt_encoder_k(input_ids, attention_mask)
        img_fea_k = nn.functional.normalize(img_fea_k, dim=1)
        txt_fea_k = nn.functional.normalize(txt_fea_k, dim=1)

        return img_fea, txt_fea, img_fea_k, txt_fea_k

    def contrastive_loss(self, stu_img, stu_txt, img_fea_k, txt_fea_k):
        """Compute the loss given pairs of image and caption embeddings
        """
        # intra-modal
        img_pos = torch.einsum('nc,nc->n', [stu_img, stu_txt]).unsqueeze(-1)
        img_neg = torch.einsum('nc,ck->nk', [stu_img, self.t_queue.clone().detach()])
        txt_pos = torch.einsum('nc,nc->n', [stu_txt, stu_img]).unsqueeze(-1)
        txt_neg = torch.einsum('nc,ck->nk', [stu_txt, self.i_queue.clone().detach()])

        # logits: Nx(1+K)
        img_logits = torch.cat([img_pos, img_neg], dim=1)
        txt_logits = torch.cat([txt_pos, txt_neg], dim=1)
        img_logits = img_logits / self.T
        txt_logits = txt_logits / self.T
        intra_labels = torch.zeros(img_logits.shape[0], dtype=torch.long).cuda()
        img_loss = nn.CrossEntropyLoss().cuda()(img_logits, intra_labels)
        txt_loss = nn.CrossEntropyLoss().cuda()(txt_logits, intra_labels)

        loss = (img_loss + txt_loss) / 2
        loss = loss * 100000

        return loss, img_loss, txt_loss


    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, i_keys, t_keys):
        batch_size = i_keys.shape[0]
        i_ptr = int(self.i_queue_ptr)
        t_ptr = int(self.t_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.i_queue[:, i_ptr:i_ptr + batch_size] = i_keys.t()  # transpose
        i_ptr = (i_ptr + batch_size) % self.K  # move pointer

        self.t_queue[:, t_ptr:t_ptr + batch_size] = t_keys.t()  # transpose
        t_ptr = (t_ptr + batch_size) % self.K  # move pointer

        self.i_queue_ptr[0] = i_ptr
        self.t_queue_ptr[0] = t_ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.img_encoder.parameters(), 
                                    self.img_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.txt_encoder.parameters(), 
                                    self.txt_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

