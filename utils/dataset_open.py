import os
from typing import List, Union
import cv2
from PIL import Image
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from bert.tokenization_bert import BertTokenizer
import random
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
# from nltk import word_tokenize, pos_tag
info = {
    'refcoco': {
        'train_seen': 35473,
        'val_seen': 3175,
        'val_unseen': 445,
        'test_seen':3200,
        'test_unseen':394
    },
    'refcoco+': {
        'train_seen': 35375,
        'val_seen': 3171,
        'val_unseen': 444,
        'test_seen':3189,
        'test_unseen':394
    },
    'refcoco_u': {
        'train_seen': 33093,
        'val_seen': 2000,
        'val_unseen': 386,
        'test_seen': 3935,
        'test_unseen': 759,
    },
    'refcoco_g': {
        'train_seen': 35105,
        'val_seen': 3923,
        'val_unseen': 760
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        #self.mask_size = [13, 26, 52]
        self.word_length = word_length
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None
    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        # mask transform
        mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.
        if self.mode == 'train':
            sent = sents[idx]
            # sentence -> vector
            img, mask, sent = self.convert(img, mask, sent, inference=False)
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            pad_mask = (word_vec != 0).float()
            return img, word_vec, mask, pad_mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[-1]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            pad_mask = (word_vec != 0).float()
            img, mask, sent = self.convert(img, mask, sent, inference=False)
            return img, word_vec, mask, pad_mask
        else:
            # sentence -> vector
            word_vecs = []
            pad_masks = []
            for sent in sents:
                word_vec = tokenize(sent, self.word_length, True).squeeze(0)
                word_vecs.append(word_vec)
                pad_mask = (word_vec != 0).float()
                pad_masks.append(pad_mask)
            img, mask, sent = self.convert(img, mask, sent, inference=True)
            return ori_img, img, word_vecs, mask, pad_masks, seg_id, sents

    def convert(self, img, mask, sent, inference=False):
        img = Image.fromarray(np.uint8(img))
        mask = Image.fromarray(np.uint8(mask), mode="P")
        img = F.resize(img, self.input_size)
        if not inference:
            mask = F.resize(mask, self.input_size, interpolation=Image.NEAREST)
        img = F.to_tensor(img)
        mask = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.int64)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask, sent

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"
