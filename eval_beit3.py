import pickle
import numpy as np
import torch
import time
from evaluation import compute_sim, i2t, t2i

def evaluate():
    img_feats_path = "knowledgevec/f30k_beit3_large_image_feats.pkl"
    txt_feats_path = "knowledgevec/f30k_beit3_large_text_feats.pkl"
    print("======>load data")
    with open(img_feats_path, 'rb') as f:
        teacher_vectors_image = pickle.load(f)
    with open(txt_feats_path, 'rb') as f:
        teacher_vectors_text = pickle.load(f)

    cap_embs = teacher_vectors_text.cpu()
    img_embs = teacher_vectors_image.cpu()
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


    return currscore

if __name__ == "__main__":
    evaluate()