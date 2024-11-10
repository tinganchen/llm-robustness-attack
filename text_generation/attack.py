import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.common as utils
from utils.options import args
from utils.load_dict import load_weight
from tensorboardX import SummaryWriter
from importlib import import_module

from data import data_loader_attack

#from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.config import GPT2Config, GPT2mConfig

from bert_score.utils import get_idf_dict
import jiwer

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#device = torch.device("cuda", args.local_rank) #device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
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
        
        model_t = import_module('model.gpt2_quant').__dict__[args.target_model](config)
        model_t = load_weight(model_t, state_dict)
        del state_dict
        
    # Data loading
    print('=> Preparing data..')
    loader = data_loader_attack.Data(args, tokenizer)
    dataset_train = loader.train_dataset.text
    
    idf_dict = get_idf_dict(dataset_train, tokenizer, nthreads=20)
    del dataset_train
 
    ## Load pretrained weights
    if args.finetuned == True:
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
    print('=> Start attack...')

    _ = adv_attack(args, loader.loader_test, model_t, tokenizer, idf_dict)
    
    print('=> Done.')
      
    
def adv_attack(args, encoded_dataset, model_t, tokenizer, idf_dict):
    
    
    '''
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print('Skipping batch as it has already been completed.')
        exit()
        '''
    #model_t = get_ddp_model(model_t)
    
    # switch to train mode
    
    num_iterations = len(encoded_dataset)
    assert args.start_index < num_iterations, 'Starting index %d is larger than dataset length %d' % (args.start_index, len(encoded_dataset))
    #end_index = min(args.start_index + int(args.attack_ratio*len(encoded_dataset)), len(encoded_dataset))
    #adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters)
    
    file_name = args.output_file
    output_file = os.path.join(args.job_dir, file_name)
    
    for idx, text in enumerate(encoded_dataset):

        #adv_log_coeffs, clean_texts, adv_texts = [], [], []
        #clean_logits = []
        #adv_logits = []
        token_errors = []

        text = text.to(device)
        label = text
        ## inference
        
        clean_logit, _ = model_t(text)  
        
        forbidden = np.zeros(len(text[0])).astype('bool')
        # set [CLS] and [SEP] tokens to forbidden
        forbidden[0] = True
        forbidden[-1] = True
        offset = 0 if args.model == 'gpt2' else 1

        forbidden_indices = np.arange(0, len(text[0]))[forbidden]
        forbidden_indices = torch.from_numpy(forbidden_indices).to(device)
        token_type_ids_batch = None
        
        #start_time = time.time()
        with torch.no_grad():
            
            embeddings = model_t.transformer.wte(torch.arange(0, tokenizer.vocab_size).long().to(device))
            orig_output = model_t.hidden_states[args.embed_layer]
            if args.constraint.startswith('bertscore'):
                if args.constraint == "bertscore_idf":
                    ref_weights = torch.FloatTensor([idf_dict[idx] for idx in text]).to(device)
                    ref_weights /= ref_weights.sum()
                else:
                    ref_weights = None
            elif args.constraint == 'cosine':
                # GPT-2 reference model uses last token embedding instead of pooling
                if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                    orig_output = orig_output[:, -1]
                else:
                    orig_output = orig_output.mean(1)
            #print(embeddings.size())
            log_coeffs = torch.zeros(len(text[0]), embeddings.size(0)) # embeddings.size(0)
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, torch.LongTensor(text[0].cpu())] = args.initial_coeff
            log_coeffs = log_coeffs.to(device)
            log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
        
        train(args, log_coeffs, embeddings, model_t, 
              text, label, forbidden_indices, optimizer, 
              idx, num_iterations)
        
        print('CLEAN TEXT')
        clean_text = tokenizer.decode(text[0, offset:(len(text[0])-offset)].tolist())
        #clean_texts.append(clean_text)
        #clean_logits.append(clean_logit)
        
        model_t.eval()
        print('ADVERSARIAL TEXT')
        with torch.no_grad():
            for j in range(args.gumbel_samples):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
                
                adv_ids = adv_ids[offset:len(adv_ids)-offset].tolist()
                adv_text = tokenizer.decode(adv_ids)
                x = tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
                token_error = wer(adv_ids, x['input_ids'][0])
                token_errors.append(token_error)
                
                adv_logit, _ = model_t(x['input_ids'].to(device))
                adv_logit = adv_logit.data
                
                last_token_logits = adv_logit[:, -1, :]
                pred = last_token_logits.argmax(dim=-1)
    
                if (pred != label).sum().item() > len(pred)*0.8 or j == args.gumbel_samples - 1:
                    #adv_texts.append(adv_text)
                    #print(adv_text)
                    #adv_logits.append(adv_logit)
                    break
                
        # remove special tokens from adv_log_coeffs
        adv_log_coeff = log_coeffs[offset:(log_coeffs.size(0)-offset), :]
        #adv_log_coeffs.append(adv_log_coeff) # size T x V
        
        '''
        print('')
        print('CLEAN LOGITS')
        print(clean_logit) # size 1 x C
        print('ADVERSARIAL LOGITS')
        print(adv_logit)   # size 1 x C
        '''
        print_logger.info("Token Error Rate: %.4f (over %d tokens)" % (sum(token_errors) / len(token_errors), len(token_errors)))
        
        
        '''
        torch.save({
            'adv_log_coeffs': adv_log_coeffs, 
            'adv_logits': torch.cat(adv_logits, 0), # size N x C
            'adv_losses': adv_loss.detach().item(),
            'adv_texts': adv_texts,
            'clean_logits': torch.cat(clean_logits, 0), 
            'clean_texts': clean_texts, 
            'entropies': entropy.detach().item(),
            'perp_losses': perp_loss.detach().item(),
            #'ref_losses': ref_losses,
            'token_error': token_errors,
        }, output_file)
        '''
        row = {'idx': idx, 'clean_text': clean_text, 'adv_text': adv_text}
        df = pd.DataFrame(list(row.items()))
        
        if idx == 0:
            df.to_csv(output_file, header=False, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)
        
        
    return 0

def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R

def train(args, log_coeffs, embeddings, model_t, text, label, forbidden_indices, optimizer, idx, num_iterations):
    
    #losses_t = utils.AverageMeter()
    losses_ppl = utils.AverageMeter()
    losses_etp = utils.AverageMeter()
    
    print('Train attack model...')
    #start = time.time()
    for i in range(args.num_iters):
        
        coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.train_batch_size, 1, 1), hard=False) # B x T x V
        inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D
        loss, logits = model_t(input_ids = text,
                               inputs_embeds = inputs_embeds, 
                               lm_labels = label)
        adv_loss = -loss
        if args.task == 'cls':
            pred = logits
            top_preds = pred.sort(descending=True)[1]
            correct = (top_preds[:, 0] == label).long()
            indices = top_preds.gather(1, correct.view(-1, 1))
            adv_loss = (pred[:, label] - pred.gather(1, indices).squeeze() + args.kappa).clamp(min=0).mean()
        

        # (log) perplexity constraint
        if args.lam_perp > 0:
            shift_logits = logits[:, :-1, :]
            shift_labels = text[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([-1, shift_logits.size(-1)]),
                shift_labels.reshape(-1),
            )   
            
            perp_loss = args.lam_perp * (-loss)
        else:
            perp_loss = torch.Tensor([0]).to(device)
            
        # Compute loss and backward
        optimizer.zero_grad()
        
        #total_loss = perp_loss
        perp_loss.backward()
        
        # Gradient step
        log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
        optimizer.step()
        optimizer.zero_grad()
        entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
        
        #losses_t.update(adv_loss.item(), text.size(0))
        losses_ppl.update(perp_loss.item(), text.size(0))
        losses_etp.update(entropy, text.size(0))
        
        '''
        if i % args.print_freq == 0:
            print_logger.info(
                    'Iter[{0}/{1}]({2}/{3}): \n'
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'PPL_loss: {ppl.val:.4f} ({ppl.avg:.4f})\n'
                    'Entropy: {etp.val:.4f} ({etp.avg:.4f})\n'.format(
                    idx, num_iterations, i, args.num_iters,
                    train_loss = losses_t,
                    ppl = losses_ppl,
                    etp = losses_etp
                    ))
            '''
            
        if i % args.print_freq == 0:
            print_logger.info(
                    'Iter[{0}/{1}]({2}/{3}): \n'
                    'PPL_loss: {ppl.val:.4f} ({ppl.avg:.4f})\n'
                    'Entropy: {etp.val:.4f} ({etp.avg:.4f})\n'.format(
                    idx, num_iterations, i, args.num_iters,
                    ppl = losses_ppl,
                    etp = losses_etp
                    ))  
            
    return losses_ppl, losses_etp


def test(args, loader_test, model_t, tokenizer):

    # switch to train mode
    model_t.eval()

    losses_t = utils.AverageMeter()
    
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
 
      
    return ppl

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


