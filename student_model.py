import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertTokenizer, ViTModel, ViTConfig
from timm import create_model
from torchvision.models import efficientnet_b0, efficientnet_b7
from transformers import BertModel, BertTokenizer
from tript_loss import TriptLoss



class Albert(nn.Module):
    def __init__(self, output_dim):
        super(Albert, self).__init__()
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

class Bert_mini(nn.Module):
    def __init__(self, output_dim):
        super(Bert_mini, self).__init__()
        model_name = "google/bert_uncased_L-4_H-256_A-4"
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
    
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
    
class Bert(nn.Module):
    def __init__(self, output_dim):
        super(Bert, self).__init__()
        model_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        feature = self.fc(pooled_output)
        return feature

class Deit(nn.Module):
    def __init__(self, output_dim):
        super(Deit, self).__init__()
        self.deit = create_model('deit_tiny_patch16_224', pretrained=True)
        self.fc = nn.Linear(self.deit.num_features, output_dim)  # 从DeiT的特征维数升至output_dim维

    def forward(self, images):
        # deit output: batch_size*num_tokens*feature_dim
        features = self.deit.forward_features(images)[:, 0, :]
        return self.fc(features)
    
    # def train(self):
    #     self.deit.train()
    #     self.fc.train()

    # def eval(self):
    #     self.deit.eval()
    #     self.fc.eval()

class Efficientb0(nn.Module):
    def __init__(self, output_dim) -> None:
        super(Efficientb0, self).__init__()
        self.efficientb0 = efficientnet_b0(pretrained=True)
        self.fc = nn.Linear(self.efficientb0.classifier[-1].out_features, output_dim)

    def forward(self, images):
        # TODO:use feature layer to output image feature, not classifier
        features = self.efficientb0(images)
        return self.fc(features)
    
class Efficientb7(nn.Module):
    def __init__(self, output_dim) -> None:
        super(Efficientb7, self).__init__()
        self.efficientb7 = efficientnet_b7(pretrained=True)
        self.fc = nn.Linear(self.efficientb7.classifier[-1].out_features, output_dim)

    def forward(self, images):
        # TODO:use feature layer to output image feature, not classifier
        features = self.efficientb7(images)
        return self.fc(features)
    


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
        self.tript_loss = TriptLoss(opt=args)

    def forward(self, images, input_ids, attention_mask):
        img_fea = self.img_encoder(images)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        txt_fea = self.txt_encoder(input_ids, attention_mask)
        img_fea = nn.functional.normalize(img_fea, dim=1)
        txt_fea = nn.functional.normalize(txt_fea, dim=1)
        return img_fea, txt_fea

    def contrastive_loss(self, stu_img, stu_txt, tea_img, tea_txt):
        """Compute the loss given pairs of image and caption embeddings
        """
        # intra-modal
        img_pos = torch.einsum('nc,nc->n', [stu_img, tea_img]).unsqueeze(-1)
        img_neg = torch.einsum('nc,ck->nk', [stu_img, self.i_queue.clone().detach()])
        txt_pos = torch.einsum('nc,nc->n', [stu_txt, tea_txt]).unsqueeze(-1)
        txt_neg = torch.einsum('nc,ck->nk', [stu_txt, self.t_queue.clone().detach()])
        # inter-modal
        i2t_pos = torch.einsum('nc,nc->n', [stu_img, tea_txt]).unsqueeze(-1)
        i2t_neg = torch.einsum('nc,ck->nk', [stu_img, self.t_queue.clone().detach()])
        t2i_pos = torch.einsum('nc,nc->n', [stu_txt, tea_img]).unsqueeze(-1)
        t2i_neg = torch.einsum('nc,ck->nk', [stu_txt, self.i_queue.clone().detach()])

        # logits: Nx(1+K)
        img_logits = torch.cat([img_pos, img_neg], dim=1)
        txt_logits = torch.cat([txt_pos, txt_neg], dim=1)
        img_logits = img_logits / self.T
        txt_logits = txt_logits / self.T
        intra_labels = torch.zeros(img_logits.shape[0], dtype=torch.long).cuda()
        img_loss = nn.CrossEntropyLoss().cuda()(img_logits, intra_labels)
        txt_loss = nn.CrossEntropyLoss().cuda()(txt_logits, intra_labels)
        
        i2t_logits = torch.cat([i2t_pos, i2t_neg], dim=1)
        t2i_logits = torch.cat([t2i_pos, t2i_neg], dim=1)
        i2t_logits = i2t_logits / self.T
        t2i_logits = t2i_logits / self.T
        inter_labels = torch.zeros(i2t_logits.shape[0], dtype=torch.long).cuda()
        i2t_loss = nn.CrossEntropyLoss().cuda()(i2t_logits, inter_labels)
        t2i_loss = nn.CrossEntropyLoss().cuda()(t2i_logits, inter_labels)

        loss = img_loss + txt_loss + i2t_loss + t2i_loss
        return loss, img_loss, txt_loss, i2t_loss, t2i_loss
    
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

