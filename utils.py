import enum
from time import time
from itsdangerous import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product


random.seed(1234)
np.random.seed(1234)
#N=10
BATCH_SIZE=128
#MAX_LENGTH=32
#POOLING='CLS'
#MODEL_NAME="bert-base-uncased"
#MODEL_NAME="princeton-nlp/unsup-simcse-bert-base-uncased"
#MODEL_NAME="../unsup_prompt"
#TXT_PATH='wiki1m_for_simcse.txt'

class Timer:
    def __init__(self) -> None:
        self.times=[time()]
    def __call__(self):
        self.times.append(time())
    def get_times(self):
        out_times=np.diff(np.array(self.times))
        
        return out_times,out_times/out_times.sum()

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    def inference(sentence_batch):
        batch = tokenizer.batch_encode_plus(
            sentence_batch,
            return_tensors="pt",
            padding=True)
        with torch.no_grad():
            outputs = model(**batch,output_hidden_states=True)
        return outputs.hidden_states,batch['attention_mask']
    return inference

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

def read_wikisent(dataset,n_lines=10):
    with open(f'../{dataset}','r') as f:
        sentences=[]
        for i in range(n_lines):
            sentences.append(f.readline())
    return sentences

def embedding2numpy(embeddings,attention_mask,pooling):
    if 'no_CLS' in pooling:
        embeddings=embeddings[:,1:]
        attention_mask=attention_mask[:,1:]
    if 'no_SEP' in pooling:
        embeddings=embeddings[:,:-1]
        attention_mask=attention_mask[:,:-1]
    if pooling=='CLS':
        embeddings=embeddings[:,1]
    elif 'avg' in pooling:
        embeddings=(embeddings* attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    elif 'none' in pooling:
        b_dim,w_dim,h_dim=embeddings.shape
        embeddings=embeddings.reshape((b_dim*w_dim,h_dim))
        attention_mask=attention_mask.reshape((b_dim*w_dim,)).bool()
        embeddings=embeddings[attention_mask]
    embeddings=embeddings.detach().numpy()
    return embeddings

def get_embedding_pairs(x):
    res=np.concatenate(x,axis=0)
    np.random.shuffle(res)
    n_tokens, embed_dim=res.shape
    if n_tokens%2!=0:
        res=res[:-1]
        n_tokens-=1
    res=res.reshape(n_tokens//2,2,embed_dim)
    return res
    
def make_embeddings(sentences,model,pooling,max_length):
    batch=[]
    token_embeddings=[]
    for sentence in tqdm(sentences):
        batch.append(' '.join(sentence.replace('\n', '').lower().split()[:max_length]))
        if len(batch) >= BATCH_SIZE:
            model_out,attention_mask=model(batch)
            model_out=list(map(lambda x: embedding2numpy(x,attention_mask,pooling),model_out))
            token_embeddings.append(model_out)
            batch = []
    if len(batch) >= 0:
        model_out,attention_mask=model(batch)
        model_out=list(map(lambda x: embedding2numpy(x,attention_mask,pooling),model_out))
        token_embeddings.append(model_out)
    token_embeddings=list(zip(*token_embeddings))
    return token_embeddings

def get_similarities(x,similarity):
    similarities=[]
    for token_pair in x:
        similarities.append(similarity(token_pair[0],token_pair[1]))
    similarities=np.array(similarities)
    return (similarities.shape[0],similarities.mean(),similarities.std())

def get_similarity_prompt(x,N,similarity):
    res=0
    x=np.concatenate(x,axis=0)
    np.random.shuffle(x)
    x=x[:N]
    x = torch.tensor(x)
    n=x.shape[0]
    for i in tqdm(range(n)):
        res=res+similarity(x[i:i+1], x).mean().item()
    return res /n

def pad_with_spaces(arg,fixed_length):
    arg=str(arg).lower()
    return ' '*(fixed_length-len(arg))+arg

def make_name(strs,fixed_lengths):
    name=[]
    for s,l in zip(strs,fixed_lengths):
        name.append(pad_with_spaces(s,l))
    return ' | '.join(name)

def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

def run_configuration(config,anisotropy_results):
    txt_name=config['dataset'].replace('.txt','')
    model_name=config['model'].split('/')[-1]
    name=f"{txt_name}|{model_name}|{config['pooling']}|{config['N']}|{config['max_length']}"
    if anisotropy_results.shape[0]==0 or (anisotropy_results['name']==name).sum()==0:
        timer=Timer()
        model=get_model(config['model'])
        timer()
        similarity=Similarity()
        sentences=read_wikisent(config['dataset'],config['N'])
        timer()
        print('Computing embeddings...')
        token_embeddings=make_embeddings(sentences,model,config['pooling'],config['max_length'])
        timer()
        print('Computing similarities...')
        per_layer_similarities=list(map(lambda x: get_similarity_prompt(x,config['N'],similarity),tqdm(token_embeddings)))
        timer()
        print(timer.get_times())
        print(per_layer_similarities)


        #name=make_name([txt_name,model_name,POOLING,N],[25,30,20,5])
        
        rows=pd.DataFrame([{
            'name':name,
            'dataset':txt_name,
            'model':model_name,
            'pooling':config['pooling'],
            'N':config['N'],
            'max_length':config['max_length'],
            'layer':i,
            'anisotropy':per_layer_similarities[i],
            'inference_time':timer.get_times()[0][2]/config['N']
        } for i in range(len(per_layer_similarities))])
        anisotropy_results=anisotropy_results.append(rows)
        print(anisotropy_results)
        anisotropy_results.to_csv('anisotropy_results.csv')
    return anisotropy_results

def grid_search(configs):
    try:
        anisotropy_results=pd.read_csv('anisotropy_results.csv',index_col=0)
    except:
        anisotropy_results=pd.DataFrame()
    configs=my_product(configs)
    print(pd.DataFrame(configs))
    for i in tqdm(range(len(configs))):
        config=configs[i]
        anisotropy_results=run_configuration(config,anisotropy_results)

def main():
    with open('config.json','r') as f:
        configs=json.load(f)
    for k,v in configs.items():
        if type(v)!=list:
            configs[k]=[v]
    grid_search(configs)


if __name__=="__main__":
    main()

    