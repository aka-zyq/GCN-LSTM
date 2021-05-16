import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
import codecs

# This file is the most important.


class MyDataset(data.Dataset):

    def __init__(self, root, caption_path, relationship_path, vocab, vocab_image, ids):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.caption_path = codecs.open(caption_path, 'r', encoding = 'utf-8')
        self.root = codecs.open(root, 'r', encoding = 'utf-8')
        self.relationship_path = codecs.open(relationship_path, 'r', encoding = 'utf-8')
        self.fcaptionLines = self.caption_path.readlines()
        self.fimageLines = self.root.readlines()
        self.frelationshipLines = self.relationship_path.readlines()
        self.vocab = vocab
        self.vocab_image = vocab
        self.ids = ids
    def __getitem__(self, index):  #Returns one data pair (image and caption).
        vocab = self.vocab
        vocab_image = self.vocab

        caption_sentence = self.fcaptionLines[index][:-1].lower()
        image_graph = self.fimageLines[index][:-1].lower()
        relationship = self.frelationshipLines[index][:-3].lower()
        
        
        #print(image.shape)   # torch.Size([3, 224, 224]) 
        # Convert caption (string) to word ids.
        tokens = caption_sentence.split('  ')
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        tokensImage = image_graph.split('  ')
        i = 0
        dic = {}
        for obj_ in tokensImage:
            dic[obj_] = i
            i += 1
        rel = relationship.split('  ')
        length = len(dic)
        matrix = np.zeros((length,length),dtype = int)
        for m in range(0,length):  # 对角矩阵，保留本结点表征
            matrix[m][m] = 1
        for r in rel:
            rs = r.split(',')
            if len(rs) == 1:     # 只有单独object，无att和relationship
                if rs[0] in dic.keys():
                    loc = dic[rs[0]]
                    if loc >= length:   # 处理数据异常时
                        pass
                    else:
                        matrix[loc][loc] = 1
                else:
                    pass
            if len(rs) == 2:
                if rs[0] in dic.keys() and rs[1] in dic.keys():
                    loc1 = dic[rs[0]]    # 有attribute或者relationship
                    loc2 = dic[rs[1]]    # 处理成对称矩阵（无向）
                    if loc1 >= length or loc2 >= length:  # 处理数据异常时
                        pass
                    else:
                        matrix[loc1][loc2] = 1
                        matrix[loc2][loc1] = 1
                else:
                    pass
        caption2 = []
        caption2.extend([vocab_image(token) for token in tokensImage])
        image = torch.Tensor(caption2)

        return image, target, matrix

    def __len__(self):
        return self.ids


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, matrixs = zip(*data)     
    lengths_image = [len(image) for image in images]
    targets_image = torch.zeros(len(images), max(lengths_image)).long()
    length_pad = max(lengths_image)    # use
    matrixs_pad = []
    for j, img in enumerate(images):
        end_img = lengths_image[j]
        targets_image[j, :end_img] = img[:end_img]
        newmatrix = np.pad(matrixs[j],((0,length_pad-len(matrixs[j])),(0,length_pad-len(matrixs[j]))),'constant',constant_values=(0,0))
        matrixs_pad.append(newmatrix)
    matrixs_pad = torch.Tensor(matrixs_pad)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]    
    return targets_image, lengths_image, targets, lengths, matrixs_pad


def get_loader(root, caption_path, relationship_path, vocab, vocab_image, batch_size, shuffle, num_workers, ids):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # caption dataset
    mydataset = MyDataset(root=root, #image_dir
                       caption_path=caption_path, #caption_path
                       relationship_path=relationship_path,
                       vocab=vocab,
                       vocab_image = vocab,
                       ids = ids)  #vocab
    
    data_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
