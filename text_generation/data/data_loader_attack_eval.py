from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
from datasets import load_dataset

import torch
import random
import warnings
from torch.utils.data.distributed import DistributedSampler

import pandas as pd

warnings.filterwarnings('ignore')

class DatasetLoader(Dataset):
    def __init__(self, args, tokenizer, src, method=None):
        super().__init__()
        
        self.args = args
        
        self.tokenizer = tokenizer
            
        file = os.path.join(args.job_dir.replace('all', f'{method}'), args.output_file)
      
        text = pd.read_csv(file, header = None)
        
        self.clean_text = text[text[0] == 'clean_text'][1]
        self.adv_text = text[text[0] == 'adv_text'][1]
        self.n_text = len(self.clean_text)
        
        text = ''
        
        if self.args.attack_ratio > 0 and self.args.attack_ratio < 1:
            n_adv_text = self.args.attack_ratio * self.n_text
            self.adv_sampled = self.n_text // n_adv_text
            
            for idx in range(len(self.clean_text)):
                if idx % self.adv_sampled == 0:
                    text += self.adv_text.iloc[idx]
                else:
                    text += self.clean_text.iloc[idx]
                
        elif self.args.attack_ratio == 0:
            text = ''.join(list(self.clean_text))
        else:
            text = ''.join(list(self.adv_text))
        
        self.corpus = tokenizer(text, return_tensors='pt').input_ids
        self.corpus = torch.concat([self.corpus, self.corpus[:, -self.args.max_seq_len+self.corpus.numel() % self.args.max_seq_len:]], dim=1)
        self.corpus = self.corpus.reshape([-1, self.args.max_seq_len])
        
    def __len__(self):
        #return self.corpus.numel() // self.args.max_seq_len
        return len(self.corpus)

    def __getitem__(self, idx):
        #return self.corpus[0, idx*self.args.max_seq_len:((idx+1)*self.args.max_seq_len)]
        return self.corpus[idx]
    

class Data:
    def __init__(self, args, tokenizer):

        
        self.loader_test = dict()
        
        if args.method == 'all':
            methods = ['wikitext', 'ptb', 'c4']
        else:
            methods = [args.method]
        
        for method in methods:
            test_dataset = DatasetLoader(args, tokenizer, 'test', method)
       
            self.loader_test[method] = DataLoader(
                        test_dataset, 
                        batch_size=args.eval_batch_size, shuffle=False, 
                        num_workers=2
                        )
      
 