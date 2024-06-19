import argparse

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
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpointv5.pth')
    parser.add_argument('--bestmodel_save_path', type=str, default='bestmodelv5.pth')
    parser.add_argument('--wandb', type=bool, action='store_true', help='use wandb')
    parser.add_argument('--wandb_project', type=str, default='KDModel')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')

    args = parser.parse_args()
    return args