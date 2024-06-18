import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from torchvision import transforms
from transformers import AlbertModel, AlbertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

class Flickr30kDataset(Dataset):
    def __init__(self, json_file, img_dir):
        """
        Args:
            json_file (string): JSON 文件的路径，包含描述和文件名。
            img_dir (string): 图像文件的目录路径。
            transform (callable, optional): 一个可选的变换函数，用于处理图像。
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-12_H-768_A-12')
        self.max_length = 30

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        description = self.annotations[idx]['description']
        # inputs = self.tokenizer(description, return_tensors="pt", max_length=self.max_length, 
        #                         padding='max_length', truncation=True)
        # 使用tokenizer对文本进行编码
        inputs = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=30,  # 设定最大长度为30
            truncation=True,
            padding='max_length',  # 填充至最大长度
            return_tensors="pt",
            return_attention_mask=True
        )

        # # 提取编码后的文本和注意力掩码
        # inputs.input_ids = inputs['input_ids'].squeeze(1)
        # inputs.attention_mask = inputs['attention_mask'].squeeze(1)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'text': inputs}

        return sample
