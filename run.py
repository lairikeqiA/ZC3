# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                        RobertaConfig, RobertaModel, RobertaTokenizer) 

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,
                 domain_label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index=index
        self.label=label
        self.domain_label=domain_label

        
def convert_examples_to_features(js,tokenizer,args):
    #source
    if "func" in js:
        code = ' '.join(js['func'].split()) 
    else:
        code = ' '.join(js['code'].split())
    #code=' '.join(js['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,int(js['index']),int(js['label']), int(js['domain_label']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data=[]
        with open(file_path) as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("label: {}".format(example.domain_label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples_0={}  
        for e in self.examples:
            if int(e.domain_label) == 0:
                if e.label not in self.label_examples_0:
                    self.label_examples_0[e.label]=[]
                self.label_examples_0[e.label].append(e)

        self.label_examples_1={} 
        for e in self.examples:
            if int(e.domain_label) == 1:
                if e.label not in self.label_examples_1:
                    self.label_examples_1[e.label]=[]
                self.label_examples_1[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        label=self.examples[i].label  
        domain_label=self.examples[i].domain_label
        index=self.examples[i].index  
        labels_0=list(self.label_examples_0)
        labels_1=list(self.label_examples_1)  #[1,2,....,64]
        if int(domain_label) == 0:
            while True:
                shuffle_example=random.sample(self.label_examples_0[label],1)[0] 
                if shuffle_example.index!=index:  #True
                    p_example=shuffle_example
                    break
        if int(domain_label) == 1:
            while True:
                shuffle_example=random.sample(self.label_examples_1[label],1)[0] 
                if shuffle_example.index!=index:  #True
                    p_example=shuffle_example
                    break

        if int(domain_label) == 0:
            labels_0.remove(label)  
            n_example=random.sample(self.label_examples_0[random.sample(labels_0,1)[0]],1)[0]  
        if int(domain_label) == 1:
            labels_1.remove(label)  
            n_example=random.sample(self.label_examples_1[random.sample(labels_1,1)[0]],1)[0] 
        
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(p_example.input_ids), torch.tensor(n_example.input_ids),
                torch.tensor(label), torch.tensor(n_example.label), torch.tensor(domain_label), torch.tensor(index))
            


class Test_TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        data = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))

        # for idx, example in enumerate(self.examples[:1]):
        #     logger.info("*** Example ***")
        #     logger.info("label: {}".format(example.label))
        #     logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
        #     logger.info("code_ids: {}".format(' '.join(map(str, example.input_ids))))
                
        self.label_examples={}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)                           
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):          
        return (torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),torch.tensor(self.examples[i].domain_label))




def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss,logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0,0.0,0.0,0,0,0
    tr_all_loss, tr_domain_loss,tr_cycle_loss,train_all_loss,train_domain_loss,train_cycle_loss,avg_all_loss,avg_domain_loss,avg_cycle_loss = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    best_acc=0.0
    cc = 0
    losses=[]
    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = train_dataloader
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            p = float(cc) / args.max_steps  
            alpha = 2. / (1. + np.exp(-100 * p)) - 1
            #alpha = 2. / (1. + np.exp(-0.01 * p)) - 1
            cc+=1

            #alpha=0.001
            
            inputs = batch[0].to(args.device)    
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            negative_labels = batch[4].to(args.device)
            domain_labels = batch[5].to(args.device)
            model.train()
            loss,domain_loss,cycle_loss,vec = model(inputs,p_inputs,n_inputs,labels,negative_labels,domain_labels,alpha)
            all_loss=loss+domain_loss+cycle_loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                domain_loss = domain_loss.mean()
                cycle_loss = cycle_loss.mean()
                all_loss = all_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                domain_loss = domain_loss / args.gradient_accumulation_steps
                cycle_loss = cycle_loss / args.gradient_accumulation_steps
                all_loss = all_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(all_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                all_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_domain_loss += domain_loss.item()
            tr_cycle_loss += cycle_loss.item()
            tr_all_loss += all_loss.item()
            tr_num+=1
            train_loss+=loss.item()
            train_domain_loss+=domain_loss.item()
            train_cycle_loss+=cycle_loss.item()
            train_all_loss+=all_loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                avg_domain_loss=tr_domain_loss
                avg_cycle_loss=tr_cycle_loss
                avg_all_loss=tr_all_loss
            avg_loss=round(train_loss/tr_num,5)
            avg_domain_loss=round(train_domain_loss/tr_num,5)
            avg_cycle_loss=round(train_cycle_loss/tr_num,5)
            avg_all_loss=round(train_all_loss/tr_num,5)
            if (step+1)% 50==0:
                logger.info("epoch {} step {} all_loss {} loss {} domain_loss {} cycle_loss {}".format(idx,step+1,avg_all_loss,avg_loss,avg_domain_loss, avg_cycle_loss))
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_all_loss
                    tr_nb=global_step

    results = evaluate(args, model, tokenizer,eval_when_training=True)
    for key, value in results.items():
        logger.info("  %s = %s", key, round(value,4))                    


eval_dataset=None
#def evaluate(args, model, tokenizer, query_data_file, candidate_data_file, eval_when_training=False):
def evaluate(args, model, tokenizer,eval_when_training=False):
    query_dataset = Test_TextDataset(tokenizer, args, args.query_data_file)
    #query_dataset = Test_TextDataset(tokenizer, args, query_data_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    candidate_dataset = Test_TextDataset(tokenizer, args, args.candidate_data_file)
    #candidate_dataset = Test_TextDataset(tokenizer, args, candidate_data_file)
    candidate_sampler = SequentialSampler(candidate_dataset)
    candidate_dataloader = DataLoader(candidate_dataset, sampler=candidate_sampler, batch_size=args.eval_batch_size, num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    model.eval()
    query_vecs = [] 
    query_labels = []
    candidate_vecs = []
    candidate_labels = []
    eval_loss = 0.0
    nb_eval_steps = 0
    # Obtain query vectors
    for batch in query_dataloader:  
        code_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        domain_label = batch[2].to(args.device)
        with torch.no_grad():
            #code_vec = model(code_inputs,code_inputs,code_inputs,label) 
            loss,domain_loss,cycle_loss,code_vec = model(code_inputs,code_inputs,code_inputs,label,label,domain_label,0.5) 
            query_vecs.append(code_vec.cpu().numpy()) 
            query_labels.append(label.cpu().numpy())
    
    # Obtain candidate vectors
    for batch in candidate_dataloader:  
        code_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        domain_label = batch[2].to(args.device)
        with torch.no_grad():
            #code_vec = model(code_inputs,code_inputs,code_inputs,label) 
            loss,domain_loss,cycle_loss,code_vec = model(code_inputs,code_inputs,code_inputs,label,label,domain_label,0.5) 
            candidate_vecs.append(code_vec.cpu().numpy()) 
            candidate_labels.append(label.cpu().numpy())
            
    model.train() 

    # Calculate cosine score
    query_vecs = np.concatenate(query_vecs,0)
    candidate_vecs = np.concatenate(candidate_vecs,0)
    query_labels = list(np.concatenate(query_labels,0))
    candidate_labels = list(np.concatenate(candidate_labels,0))
    scores = np.matmul(query_vecs,candidate_vecs.T)
    
    # Calculate MAP score
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP=[]
    results = {}
    for i in range(scores.shape[0]):
        cont=0
        label=int(query_labels[i])
        Avep = []
        for j,index in enumerate(list(sort_ids[i])):
            if  int(candidate_labels[index])==label:
                Avep.append((len(Avep)+1)/(j+1))
        if len(Avep)!=0:
            MAP.append(sum(Avep)/len(Avep))
   
    result = {
        "qc_eval_map":float(np.mean(MAP))
    }


    cq_scores = np.matmul(candidate_vecs,query_vecs.T)
    
    # Calculate MAP score
    cq_sort_ids = np.argsort(cq_scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP=[]
    results = {}
    for i in range(cq_scores.shape[0]):
        cont=0
        label=int(candidate_labels[i])
        Avep = []
        for j,index in enumerate(list(cq_sort_ids[i])):
            if  int(query_labels[index])==label:
                Avep.append((len(Avep)+1)/(j+1))
        if len(Avep)!=0:
            MAP.append(sum(Avep)/len(Avep))
   
    result["cq_eval_map"]=float(np.mean(MAP))
    
    return result




def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)    
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        negative_labels = batch[4].to(args.device)
        domain_label = batch[5].to(args.device)
        with torch.no_grad():
            lm_loss,domain_loss,vec = model(inputs,p_inputs,n_inputs,label,negative_labels, domain_label,alpha=0.5)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs=np.concatenate(vecs,0)
    labels=np.concatenate(labels,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(vecs, vecs.T)
    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append(sum(Avep) / dic[label])

    result = {
        "eval_loss": float(perplexity),
        "eval_map": float(np.mean(MAP))
    }


    scores=np.matmul(vecs,vecs.T)
    for i in range(scores.shape[0]):
        scores[i,i]=-1000000
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    indexs=[]
    for example in eval_dataset.examples:
        indexs.append(example.index)
    with open(os.path.join(args.output_dir,"predictions.jsonl"),'w') as f:
        for index,sort_id in zip(indexs,sort_ids):
            js={}
            js['index']=index
            js['answers']=[]
            for idx in sort_id[:499]:
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js)+'\n')

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--query_data_file", default=None, type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--candidate_data_file", default=None, type=str, required=False,
                        help="The input training data file (a json file).")  
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropouts.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    

    args = parser.parse_args()


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                              cache_dir=args.cache_dir if args.cache_dir else None)


    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    print('len tokenizer', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))  # add special token

    model=Model(model,config,tokenizer,args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:

        params = list(model.named_parameters())
        print()
        print(params.__len__())
        print('params0\n', params[0])


        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



if __name__ == "__main__":
    main()



