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

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}
_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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
    l_mask = [0] * context_length
    result = [0] * context_length

    tokens = _tokenizer.encode(text=texts, add_special_tokens=True)
    tokens = tokens[:context_length]
    result[:len(tokens)] = tokens
    l_mask[:len(tokens)] = [1]*len(tokens)

    result = torch.tensor(result).unsqueeze(0)
    l_mask = torch.tensor(l_mask).unsqueeze(0)
    return result, l_mask


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
        # self.coco_transforms = make_coco_transforms(mode, cautious=False)

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
            word_vec, pad_mask = tokenize(sent, self.word_length, True)
            return img, word_vec, mask, pad_mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[-1]
            word_vec, pad_mask = tokenize(sent, self.word_length, True)
            img, mask, sent = self.convert(img, mask, sent, inference=False)
            return img, word_vec, mask, pad_mask
        else:
            # sentence -> vector
            word_vecs = []
            pad_masks = []
            for sent in sents:
                word_vec, pad_mask = tokenize(sent, self.word_length, True)
                word_vecs.append(word_vec)
                pad_masks.append(pad_mask)
            img, mask, sent = self.convert(img, mask, sent, inference=True)
            return ori_img, img, word_vecs, mask, pad_masks, seg_id, sents, 

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


