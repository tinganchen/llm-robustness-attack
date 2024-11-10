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
    def __init__(self, args, tokenizer, src):
        super().__init__()
        
        self.args = args
        
        text = '' 
    
        method = args.method
                
        file = os.path.join(args.data_path, f'{method}/{method}_{src}.txt')
            
        with open(file, 'r') as f:
            t = f.readlines()
        
        text += "\n\n".join(t)
        
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
        
        self.train_dataset = DatasetLoader(args, tokenizer, 'train')
   
        self.loader_train = DataLoader(
                    self.train_dataset, 
                    batch_size=args.train_batch_size, shuffle=True, 
                    num_workers=2
                    )
        
        val_dataset = DatasetLoader(args, tokenizer, 'validation')
   
        self.loader_validation = DataLoader(
                    val_dataset, 
                    batch_size=args.eval_batch_size, shuffle=False, 
                    num_workers=2
                    )
        
        
        test_dataset = DatasetLoader(args, tokenizer, 'test')
   
        self.loader_test = DataLoader(
                    test_dataset, 
                    batch_size=args.eval_batch_size, shuffle=False, 
                    num_workers=2
                    )
      
 