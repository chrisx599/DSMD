from student_train import evaluate, get_args
from student_model import KDModel
from student_dataset import Flickr30kDataset, DataLoader
import os
import pickle   
from evaluation import compute_sim, i2t, t2i
import time

args = get_args()
model = KDModel(768)
model.cuda()
val_dataset = Flickr30kDataset(os.path.join(args.dataset, 'val.json')
                                   , os.path.join(args.dataset, 'flickr30k-images'))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size
                        , shuffle=False, num_workers=args.num_workers)
model.eval()
evaluate(model, val_loader)
print('666')



