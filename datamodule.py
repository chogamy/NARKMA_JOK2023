import argparse
from dataclasses import dataclass

import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from splitter_train import split

from mytokenizer import MyTokenizer

@dataclass
class DataCollatorForSeq2Seq:
    src_tok : MyTokenizer
    morph_tok : MyTokenizer
    tag_tok : MyTokenizer
    max_len : int = 200
    label_pad_token_id : int = -100

    def __call__(self, features):

        # input ids, morph_labels, tag_labels
        for feature in features:
            feature['len_labels'] = self.make_len_labels(feature['input_ids'], feature['morph_labels'])
            feature['input_ids'], feature['attention_mask'] = self.make_encoder_input(feature['input_ids'])
            feature['morph_input_ids'], feature['tag_input_ids'], feature['dec_attention_mask'] = self.make_decoder_input(feature['morph_labels'], feature['tag_labels'])
            feature['morph_labels'], feature['tag_labels'] = self.make_dec_labels(feature['morph_labels'], feature['tag_labels'])
        
        batch = {
            "input_ids": torch.LongTensor(np.stack([feature['input_ids'] for feature in features])),
            "attention_mask": torch.LongTensor(np.stack([feature['attention_mask'] for feature in features])),
            "morph_input_ids": torch.LongTensor(np.stack([feature['morph_input_ids'] for feature in features])),
            "tag_input_ids": torch.LongTensor(np.stack([feature['tag_input_ids'] for feature in features])),
            "dec_attention_mask": torch.LongTensor(np.stack([feature['dec_attention_mask'] for feature in features])),
            "morph_labels": torch.LongTensor(np.stack([feature['morph_labels'] for feature in features])),
            "tag_labels": torch.LongTensor(np.stack([feature['tag_labels'] for feature in features])),
            "len_labels": torch.LongTensor(np.stack([feature['len_labels'] for feature in features]))
        }

        return batch

    def make_dec_labels(self, morph_labels, tag_labels):
        morph_labels = np.concatenate([
            morph_labels,
            [self.label_pad_token_id] * (self.max_len - len(morph_labels))
        ])
        tag_labels = np.concatenate([
            tag_labels,
            [self.label_pad_token_id] * (self.max_len - len(tag_labels))
        ])

        return morph_labels, tag_labels

    def make_decoder_input(self, morph_labels, tag_labels):
        assert len(morph_labels) == len(tag_labels), f"leng diff"

        morph_input_ids = np.where(morph_labels==self.morph_tok.index(" "),
                                morph_labels,
                                self.morph_tok.index("<mask>")
                                )
        tag_input_ids = np.where(tag_labels==self.tag_tok.index("/O"),
                                tag_labels,
                                self.tag_tok.index("<mask>")
                                )

        dec_attention_mask = np.concatenate([
            [1] * len(morph_input_ids),
            [0] * (self.max_len - len(morph_input_ids))
        ])

        assert self.morph_tok.index("<pad>") == self.tag_tok.index("<pad>")
        morph_input_ids = np.concatenate([
            morph_input_ids,
            [self.morph_tok.index("<pad>")] * (self.max_len - len(morph_input_ids))
        ])

        tag_input_ids = np.concatenate([
            tag_input_ids,
            [self.tag_tok.index("<pad>")] * (self.max_len - len(tag_input_ids))
        ])

        return morph_input_ids, tag_input_ids, dec_attention_mask
        
    def make_encoder_input(self, input_ids):
        attention_mask = np.concatenate([
            [1] * len(input_ids),
            [0] * (self.max_len - len(input_ids))
        ])
        input_ids = np.concatenate([
            input_ids,
            [self.src_tok.pad_index] * (self.max_len - len(input_ids))
        ])
        return input_ids, attention_mask

    def make_len_labels(self, input_ids, morph_labels):

        src_eojs = []
        src_eoj = []
        for i in input_ids:
            if i == self.src_tok.index(" "):
                src_eojs.append(src_eoj)
                src_eoj=[]
            else :
                src_eoj.append(i)
        if len(src_eoj) != 0:
            src_eojs.append(src_eoj)
            src_eoj = []
        
        tgt_eojs = []
        tgt_eoj = []
        for i in morph_labels:
            if i == self.morph_tok.index(" "):
                tgt_eojs.append(tgt_eoj)
                tgt_eoj = []
            else:
                tgt_eoj.append(i)

        if len(tgt_eoj) != 0:
            tgt_eojs.append(tgt_eoj)
            tgt_eoj = []
        
        assert len(src_eojs)==len(tgt_eojs), f"{len(src_eojs)} {len(tgt_eojs)}"

        len_labels = []
        cur_len = 0
        for src_eoj, tgt_eoj in zip(src_eojs, tgt_eojs):
            cur_len = len(tgt_eoj)
            for _ in range(0, len(src_eoj)):
                len_labels.append(cur_len)
            len_labels.append(0)
        
        if input_ids[-1] == self.src_tok.index(" "):
            pass
        else:
            len_labels = len_labels[:-1]
        
        assert len(len_labels) == len(input_ids), f"{len(len_labels)} {len(input_ids)} \n{len_labels} \n{self.src_tok.decode(input_ids, False)}"

        len_labels = np.concatenate([
            len_labels,
            [self.label_pad_token_id] * (self.max_len - len(len_labels))
        ])
        assert len(len_labels) == self.max_len, f" len of len label is under max_len"
        return len_labels
        
class KMADataset(Dataset):
    def __init__(self, filepath, src_tok, morph_tok, tag_tok, max_len, ignore_index=-100) -> None:
        self.filepath = filepath

        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok

        self.max_len = max_len
        self.srcs, self.morphs, self.tags = self.load_data()

        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.srcs)

    
    def __getitem__(self, index):
        
        # SRC
        src_sent = self.srcs[index]
        src_tokens = list(src_sent) 
        input_ids = self.src_tok.encode(src_tokens)

        # Morph
        morph_sent = self.morphs[index]
        morph_tokens = list(morph_sent)
        morph_labels = self.morph_tok.encode(morph_tokens)

        # Tag
        tag_tokens = self.tags[index].strip().split(" ")
        tag_labels = self.tag_tok.encode(tag_tokens)

        assert len(morph_labels) == len(tag_labels), f"morph and tag len diff"

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'morph_labels': np.array(morph_labels, dtype=np.int_),
                'tag_labels': np.array(tag_labels, dtype=np.int_),
                }

    def load_data(self):
        srcs = []
        morphs = []
        tags = []

        src_f = open(self.filepath + "_src.txt", 'r', encoding="UTF-8-sig")
        morph_f = open(self.filepath + "_morph.txt", 'r', encoding="UTF-8-sig")
        tag_f = open(self.filepath + "_tag.txt", 'r', encoding="UTF-8-sig")

        for src, morph, tag in zip(src_f, morph_f, tag_f):
            src_bufs, morph_bufs, tag_bufs = split(src.strip(), morph.strip(), tag.strip(), self.max_len)

            for src_buf, morph_buf, tag_buf in zip(src_bufs, morph_bufs, tag_bufs):
                srcs.append(src_buf)
                morphs.append(morph_buf)
                tags.append(tag_buf)

        print(len(srcs))    
        assert len(srcs) == len(morphs) == len(tags), "length different"
        return srcs, morphs, tags

class KMAModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, src_tok, morph_tok, tag_tok, max_len, batch_size=8, num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok
        self.max_len = max_len

        self.num_workers = num_workers

        self.data_collator = DataCollatorForSeq2Seq(src_tok=self.src_tok, morph_tok=self.morph_tok, tag_tok=self.tag_tok, max_len=self.max_len)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KMADataset(self.train_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)
        self.valid = KMADataset(self.valid_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train, collate_fn=self.data_collator,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid, collate_fn=self.data_collator,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val
