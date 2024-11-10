import os
import torch
import torch.nn as nn
import numpy as np
import utils.common as utils
from utils.options import args
from utils.load_dict import load_weight
from tensorboardX import SummaryWriter
from importlib import import_module

from data import data_loader_attack_eval

#from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.config import GPT2Config, GPT2mConfig

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import math

import warnings
warnings.filterwarnings('ignore')

#device = torch.device("cuda", args.local_rank) #device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, f"logger_eval_{args.bitW}bit.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


# Task: https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7
'''
if torch.cuda.device_count() > 1:
    dist.init_process_group(backend='nccl')
    dist.barrier()
    world_size = dist.get_world_size()
    device = args.local_rank
else:
    device = torch.device(f"cuda:{args.gpus[0]}")
    '''
device = torch.device(f"cuda:{args.gpus[0]}")    

def main():
    
    best_ppl = math.inf
    
    # Create model
    print('=> Building model...')
    
    ## Load pre-trained model (weights)
    if 'opt' in args.model:
        if args.bitW == 32 and args.abitW == 32:
            opt_class = import_module('model.opt').__dict__[args.target_model](args)
        else:
            opt_class = import_module('model.opt_quant').__dict__[args.target_model](args)
        tokenizer = opt_class.tokenizer
        model_t = opt_class.model
    
    elif 'gpt2' in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(f'{args.model}') #GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained(f'{args.model}') 
        state_dict = model.state_dict()
        del model
        
        ## Create trained model (architecture)
        if args.model == 'gpt2':
            config = GPT2Config()
        elif args.model == 'gpt2-medium':
            config = GPT2mConfig()
        
        if args.bitW == 32 and args.abitW == 32:
            model_t = import_module('model.gpt2').__dict__[args.target_model](config)
        else:
            model_t = import_module('model.gpt2_quant').__dict__[args.target_model](config)
        model_t = load_weight(model_t, state_dict)
        del state_dict
        
    # Data loading
    print('=> Preparing data..')
    loader = data_loader_attack_eval.Data(args, tokenizer)
 
    ## Load pretrained weights
    if args.finetuned:
        ckpt = torch.load(os.path.join(args.source_file, 'checkpoint/model_best.pt'), map_location = device)
        state_dict = ckpt['state_dict']
        
        state_dict_t = dict()
        
        for k, v in model_t.state_dict().items():
            if k in state_dict:
                state_dict_t[k] = state_dict[k]
            else:
                state_dict_t[k] = v
        model_t.load_state_dict(state_dict_t)
        model_t = model_t.to(device)
        
        del ckpt, state_dict, state_dict_t
        print('=> Finish Loading.')
    
    model_t = model_t.to(device)
   
    # inference
    print('=> Start inference...')

    methods, ppls = eval_ppl(args, loader.loader_test, model_t, tokenizer)
    
    print_logger.info(f"Attack Ratio: {args.attack_ratio} ===>")
    
    for i in range(len(methods)):
        print_logger.info(f"{methods[i]} Best @ppl: {ppls[i]:.2f}\n")
        
    print('=> Done.')
        
    
def eval_ppl(args, loaders_test, model_t, tokenizer):

    # switch to train mode
    model_t.eval()
    
    methods = []
    ppls = []
    
    for k, loader_test in loaders_test.items():
        losses_t = utils.AverageMeter()
        print(f'==> Method {k}...')
        
        for i, text in enumerate(loader_test, 1):
            text = text.to(device)
            
            if 'opt' in args.model:
                decoder = get_ddp_model(model_t.model.decoder)
                
                outputs = decoder(text)    
                hidden_states = outputs[0]
                
                #print('Calculate logits..')
                lm_head = get_ddp_model(model_t.lm_head)
                logits = lm_head(hidden_states) 
            else:
                ## inference
                model_t = get_ddp_model(model_t)
                logits, _ = model_t(text)  
            
            shift_logits = logits[:, :-1, :]
            shift_labels = text[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([-1, shift_logits.size(-1)]),
                shift_labels.reshape(-1),
            )                
    
            losses_t.update(loss.item(), text.size(0))
            
        ## evaluate
        ppl = math.exp(losses_t.avg)  
        ppls.append(ppl)
        methods.append(k)
      
    return methods, ppls

def get_ddp_model(model):
    '''
    if torch.cuda.device_count() > 1:
        # parallel
        model = DDP(model, device_ids=[args.local_rank], 
                    #output_device=args.local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
        '''
    return model
    
if __name__ == '__main__':
    main()


