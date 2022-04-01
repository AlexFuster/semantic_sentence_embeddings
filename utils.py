from time import time
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from itertools import product
from nltk.corpus import stopwords


#N=10
BATCH_SIZE=128
STOPWORDS=stopwords.words('english')
#OUT_PATH='anisotropy_results.csv'
OUT_PATH='anisotropy_results.csv'
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

class Evaluator():
    def __init__(self,config,results) -> None:
        self.config=config
        self.anisotropy_results=results
        self.similarity=Similarity()
        with open('token_freqs.json','r') as f:
            self.dataset_frequencies=json.load(f)
        self.config['freq_bias']=False

    def get_model(self):
        model_name=self.config['model']
        print(f'getting model from {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        def inference(sentence_batch):
            batch = tokenizer.batch_encode_plus(
                sentence_batch,
                return_tensors="pt",
                padding=True)
            with torch.no_grad():
                outputs = model(**batch,output_hidden_states=True)
            return outputs.hidden_states,batch['attention_mask'], list(map(tokenizer.convert_ids_to_tokens,batch.input_ids))
        return inference

    def read_wikisent(self):
        with open(f"../{self.config['dataset']}",'r') as f:
            sentences=[]
            for i in range(self.config['N']):
                sentences.append(f.readline())
        return sentences

    def remove_stopword_embeddings(self,batch_tokens,attention_mask):
        if self.config['no_stop']:
            stopwords_mask=np.ones(attention_mask.shape)
            for i,sent in enumerate(batch_tokens):
                for j,token in enumerate(sent):
                    if token in STOPWORDS:
                        stopwords_mask[i,j]=0
            attention_mask=attention_mask*torch.from_numpy(stopwords_mask)
        return attention_mask

    def embedding2numpy(self,embeddings,attention_mask):
        if self.config['pooling']=='CLS':
            embeddings=embeddings[:,1]
        elif 'avg' in self.config['pooling']:
            embeddings=(embeddings* attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif 'none' in self.config['pooling']:
            b_dim,w_dim,h_dim=embeddings.shape
            embeddings=embeddings.reshape((b_dim*w_dim,h_dim))
            attention_mask=attention_mask.reshape((b_dim*w_dim,)).bool()
            embeddings=embeddings[attention_mask]
        embeddings=embeddings.detach().numpy()
        return embeddings

    def compute_freqs(self,batch_tokens,attention_mask):
        if self.config['freq_bias']:
            freqs=[]
            for i in range(len(batch_tokens)):
                for j in range(len(batch_tokens[i])):
                    if attention_mask[i,j]==1:
                        freqs.append(self.dataset_frequencies[batch_tokens[i][j]])
            return np.array(freqs)

    def process_batch(self,token_embeddings,token_freqs,batch):
        model_out,attention_mask,batch_tokens=self.model(batch)
        if 'no_CLS' in self.config['pooling']:
            attention_mask[:,0]=0
        if 'no_SEP' in self.config['pooling']:
            sep_mask=F.pad(attention_mask,[0,1])
            sep_mask=torch.minimum(sep_mask[:,1:]-sep_mask[:,:-1],torch.Tensor([0]))
            attention_mask=attention_mask+sep_mask

        attention_mask=self.remove_stopword_embeddings(batch_tokens,attention_mask)

        model_out=list(map(lambda x: self.embedding2numpy(x,attention_mask),model_out))
        token_embeddings.append(model_out)
        token_freqs.append(self.compute_freqs(batch_tokens,attention_mask))

    def make_embeddings(self,sentences):
        batch=[]
        token_embeddings=[]
        token_freqs=[]
        for sentence in tqdm(sentences):
            if not self.config['cased']:
                sentence=sentence.lower()
            batch.append(' '.join(sentence.replace('\n', '').split()[:self.config['max_length']]))
            if len(batch) >= BATCH_SIZE:
                self.process_batch(token_embeddings,token_freqs,batch)
                batch = []
        if len(batch) >= 0:
            self.process_batch(token_embeddings,token_freqs,batch)
        token_embeddings=list(zip(*token_embeddings))
        return token_embeddings, token_freqs

    def get_similarity_prompt(self,x,freqs):
        res_isot=0
        x=np.concatenate(x,axis=0)
        np.random.seed(1234)
        np.random.shuffle(x)
        x=x[:self.config['N']]
        x = torch.tensor(x)
        n=x.shape[0]
        results=np.zeros((n**2,2))

        for i in tqdm(range(n)):
            aux_sim=self.similarity(x[i:i+1], x).detach().numpy()
            res_isot=res_isot+aux_sim.mean()
            results[i*n:(i+1)*n,0]=aux_sim

        if self.config['freq_bias']:
            freqs=np.concatenate(freqs,axis=0)
            for i in tqdm(range(n)):
                aux_freq_dif=abs(freqs[i:i+1]-freqs)
                results[i*n:(i+1)*n,1]=aux_freq_dif
            
        return res_isot/n

    def run_configuration(self):
        txt_name=self.config['dataset'].replace('.txt','')
        model_name=self.config['model'].split('/')[-1]
        name=f"{txt_name}|{model_name}|{self.config['pooling']}|{self.config['N']}|{self.config['max_length']}"
        combination_config={
            'dataset':txt_name,
            'model':model_name,
            'pooling':self.config['pooling'],
            'N':self.config['N'],
            'max_length':self.config['max_length'],
            'cased':self.config['cased'],
            'no_stop':self.config['no_stop']
        }

        query=[]
        for k,v in combination_config.items():
            if type(v)==str:
                query.append(f"{k} == '{v}'")
            else:
                query.append(f"{k} == {v}")

        query=' & '.join(query)

        if True:#self.anisotropy_results.shape[0]==0 or self.anisotropy_results.query(query).shape[0]==0:
            timer=Timer()
            self.model=self.get_model()
            timer()
            sentences=self.read_wikisent()
            timer()
            print('Computing embeddings...')
            token_embeddings, token_freqs=self.make_embeddings(sentences)
            timer()
            print('Computing similarities...')
            per_layer_similarities=list(map(lambda x: self.get_similarity_prompt(x,token_freqs),tqdm(token_embeddings)))
            timer()
            print(timer.get_times())
            print(per_layer_similarities)

            #name=make_name([txt_name,model_name,POOLING,N],[25,30,20,5])
            
            rows=pd.DataFrame([dict(
                **combination_config,
                name=name,
                layer=i,
                anisotropy=per_layer_similarities[i],
                inference_time=timer.get_times()[0][2]/self.config['N'],
                
            ) for i in range(len(per_layer_similarities))])
            self.anisotropy_results=self.anisotropy_results.append(rows)
            print(self.anisotropy_results)
            self.anisotropy_results.to_csv(OUT_PATH)
        return self.anisotropy_results



"""
def get_embedding_pairs(x):
    res=np.concatenate(x,axis=0)
    np.random.shuffle(res)
    n_tokens, embed_dim=res.shape
    if n_tokens%2!=0:
        res=res[:-1]
        n_tokens-=1
    res=res.reshape(n_tokens//2,2,embed_dim)
    return res

def get_similarities(x,similarity):
    similarities=[]
    for token_pair in x:
        similarities.append(similarity(token_pair[0],token_pair[1]))
    similarities=np.array(similarities)
    return (similarities.shape[0],similarities.mean(),similarities.std())

def pad_with_spaces(arg,fixed_length):
    arg=str(arg).lower()
    return ' '*(fixed_length-len(arg))+arg

def make_name(strs,fixed_lengths):
    name=[]
    for s,l in zip(strs,fixed_lengths):
        name.append(pad_with_spaces(s,l))
    return ' | '.join(name)
"""

def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

def grid_search(configs):
    try:
        anisotropy_results=pd.read_csv(OUT_PATH,index_col=0)
    except:
        anisotropy_results=pd.DataFrame()
    configs=my_product(configs)
    print(pd.DataFrame(configs))
    for i in tqdm(range(len(configs))):
        config=configs[i]
        evaluator=Evaluator(config,anisotropy_results)
        anisotropy_results=evaluator.run_configuration()

def main(config_path):
    with open(config_path,'r') as f:
        configs=json.load(f)
    for k,v in configs.items():
        if type(v)!=list:
            configs[k]=[v]
    grid_search(configs)


if __name__=="__main__":
    if len(sys.argv)>1:
        config_path=sys.argv[1]
    else:
        config_path='config.json'
    main(config_path)

    