import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = 't_0'

'''
tmux, index

'''

WARMUP_STEPS = 5000
MAX_SEQ_LEN = 1024

FINETUNED = True
TEST_ONLY = False
STAGE = 'second'
TASK = 'gen' # 'cls'
CONSTAINT = 'bertscore_idf' # 'cosine'

MODEL = 'gpt2'# 'facebook/opt-125m' 
METHOD = 'all' # 'wikitext', 'ptb', 'c4', 'lambada'

# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Dataset to train')


#parser.add_argument('--data_path', type = str, default = '/media/ta/e9cf3417-0c3e-4e6a-b63c-4401fabeabc8/ta/test_gpt_datasets', help = 'The directory where the input data is stored.')
parser.add_argument('--data_path', type = str, default = '/home/tachen/dataset/test_gpt_datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_path', type = str, default = '/home/ta/research/dataset/test_gpt_datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_path', type = str, default = '/home/tachen/research', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_path', type = str, default = '/home/annachen/research/datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_path', type = str, default = '/workspace/datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--cache_dir', type = str, default = 'data/cache_dir/', help = 'The directory where the input data is stored.')
parser.add_argument('--train_data_dir', type = str, default = f'{METHOD}/{METHOD}_train.txt', help = 'The directory where the input data is stored.')
parser.add_argument('--validation_data_dir', type = str, default = f'{METHOD}/{METHOD}_validation.txt', help = 'The directory where the input data is stored.')
parser.add_argument('--test_data_dir', type = str, default = f'{METHOD}/{METHOD}_test.txt', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'../../llm_experiment/{MODEL.split("/")[-1]}/{METHOD}/{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--output_file', type = str, default = 'gen_output.csv', help = 'The directory where the input data is stored.')
#parser.add_argument("--local_rank", default=os.environ['LOCAL_RANK'], type=int)
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--finetuned', type = str, default = FINETUNED, help = 'Whether to use a finetuned model')
parser.add_argument('--test_only', type = str, default = TEST_ONLY, help = 'Test or not')
parser.add_argument('--stage', type = str, default = STAGE, help = 'Load pruned model')
parser.add_argument('--method', type = str, default = METHOD, help = 'Load pruned model')

parser.add_argument('--source_dir', type = str, default = '', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = f'/home/ta/research/experiment/llm_experiment/{MODEL.split("/")[-1]}/all/t_0/', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None

# Attack
parser.add_argument("--start_index", default=0, type=int, help="starting sample index")
parser.add_argument("--attack_ratio", default=0.1, type=float, help="proportion of samples to attack")
parser.add_argument("--gumbel_samples", default=1, type=int, help="number of gumbel samples; if 0, use argmax")
parser.add_argument('--constraint', type = str, default = CONSTAINT, help = 'Attack constraint.') # None

parser.add_argument("--embed_layer", default=-1, type=int, help="which layer of LM to extract embeddings from")
parser.add_argument("--initial_coeff", default=15, type=int, help="initial log coefficients")
parser.add_argument( '--task',  type = str, default = TASK, help = 'Text generation of classification.')
parser.add_argument("--lam_perp", default=1, type=float, help="(log) perplexity regularizer")

## Training
parser.add_argument('--bitW', type = int, default = 8, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = 8, help = 'Quantized bitwidth.') # None
parser.add_argument('--img_size', type = int, default = 32, help = 'Quantized bitwidth.') # None

parser.add_argument('--target_model', type = str, default = 'GPT2LMHeadModel', help = 'The target model.')
parser.add_argument('--model', type = str, default = MODEL, help = 'The target model.')

parser.add_argument('--source_model', type = str, default = 'resnet_20', help = 'The baseline model.') 
parser.add_argument('--num_epochs', type = int, default = 10, help = 'The num of epochs to train.') # 100
parser.add_argument("--num_iters", default=100, type=int, help="number of epochs to train for")# 1. train: default = 100


parser.add_argument('--train_batch_size', type = int, default = 1, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 1, help = 'Batch size for validation.')
#parser.add_argument('--batch_size', type = int, default = 1, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 3e-5)
# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.1)

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [80, 150])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'The weight decay of loss.')

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
parser.add_argument('--warmup_steps', type = int, default = WARMUP_STEPS, help = 'Modify the approximated slope of step function.')
parser.add_argument('--max_seq_len', type = int, default = MAX_SEQ_LEN, help = 'Scale the sigmoid function.')

## Status
parser.add_argument('--print_freq', type = int, default = 50, help = 'The frequency to print loss.')

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

