import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    def __init__(self, t=0.05):
        super(ContrastiveLoss, self).__init__()
        self.T = t

    def forward(self, stu_img, stu_txt, tea_img, tea_txt):
            """Compute the loss given pairs of image and caption embeddings
            """
            # intra-modal
            img_pos = torch.einsum('nc,nc->n', [stu_img, tea_img]).unsqueeze(-1)
            img_neg = torch.einsum('nc,ck->nk', [stu_img, self.i_queue.clone().detach()])
            txt_pos = torch.einsum('nc,nc->n', [stu_txt, tea_txt]).unsqueeze(-1)
            txt_neg = torch.einsum('nc,ck->nk', [stu_txt, self.t_queue.clone().detach()])

            # logits: Nx(1+K)
            img_logits = torch.cat([img_pos, img_neg], dim=1)
            txt_logits = torch.cat([txt_pos, txt_neg], dim=1)
            img_logits = img_logits / self.T
            txt_logits = txt_logits / self.T
            intra_labels = torch.zeros(img_logits.shape[0], dtype=torch.long).cuda()
            img_loss = nn.CrossEntropyLoss().cuda()(img_logits, intra_labels)
            txt_loss = nn.CrossEntropyLoss().cuda()(txt_logits, intra_labels)

            loss = img_loss + txt_loss
            return loss, img_loss, txt_loss


class TriptLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt):
        super(TriptLoss, self).__init__()
        self.margin = opt.margin
        self.max_violation = opt.max_violation

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
