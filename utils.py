from time import time
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from itertools import product
from nltk.corpus import stopwords
from scipy.stats import spearmanr
import csv

from metrics import Cos, Euclidean

#N=10
BATCH_SIZE=128
STOPWORDS=stopwords.words('english')
#TASK='STS'
TASK='isotropy'
OUT_PATH=f'{TASK}_results.csv'
pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', ',', '¿', '-', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
TASK='bias'
#TASK='case_bias'
#TASK='subword_bias'

class Timer:
    def __init__(self) -> None:
        self.times=[time()]
    def __call__(self):
        self.times.append(time())
    def get_times(self):
        out_times=np.diff(np.array(self.times))
        
        return out_times,out_times/out_times.sum()

def deterministic_shuffle(x):
    np.random.seed(1234)
    np.random.shuffle(x)

def is_stopword(token):
    return token in STOPWORDS

def is_subword(token):
    return token.startswith('#') or token in pun_remove_set

class Evaluator():
    def __init__(self,config,results) -> None:
        self.config=config
        self.results=results
        if config['metric']=='cosine':
            self.similarity=Cos()
        elif config['metric']=='euclidean':
            self.similarity=Euclidean()
        else: raise ValueError('Non existent metric')

        with open('token_freqs_2.json','r') as f:
            self.dataset_frequencies=json.load(f)

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

    def remove_sto_sub_embeddings(self,batch_tokens,attention_mask):
        if self.config['no_stop'] or self.config['no_sub']:
            aux_mask=np.ones(attention_mask.shape)
            for i,sent in enumerate(batch_tokens):
                for j,token in enumerate(sent):
                    if (self.config['no_stop'] and is_stopword(token)) or (self.config['no_sub'] and is_subword(token)):
                        aux_mask[i,j]=0
            attention_mask=attention_mask*torch.from_numpy(aux_mask)
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

    def compute_metadata(self,tokens_metadata,batch_tokens,attention_mask):
        if 'bias' in TASK:
            for i in range(len(batch_tokens)):
                for j in range(len(batch_tokens[i])):
                    if attention_mask[i,j]==1:
                        tokens_metadata['stopword_flag'].append(is_stopword(batch_tokens[i][j]))
                        tokens_metadata['lowcase_flag'].append(batch_tokens[i][j].islower())
                        tokens_metadata['subword_flag'].append(is_subword(batch_tokens[i][j]))

    def process_batch(self,token_embeddings,tokens_metadata,batch):
        model_out,attention_mask,batch_tokens=self.model(batch)
        if self.config['no_cls']:
            attention_mask[:,0]=0
        if self.config['no_sep']:
            sep_mask=F.pad(attention_mask,[0,1])
            sep_mask=torch.minimum(sep_mask[:,1:]-sep_mask[:,:-1],torch.Tensor([0]))
            attention_mask=attention_mask+sep_mask
        attention_mask=self.remove_sto_sub_embeddings(batch_tokens,attention_mask)
        #has_tokens_left=torch.sum(attention_mask,dim=1)>0
        #attention_mask=attention_mask[has_tokens_left]
        #batch_tokens=list(list(zip(*list(filter(lambda x:x[1],zip(batch_tokens,has_tokens_left)))))[0])

        self.compute_metadata(tokens_metadata,batch_tokens,attention_mask)

        model_out=list(map(lambda x: self.embedding2numpy(x,attention_mask),model_out))
        token_embeddings.append(model_out)

    def make_embeddings(self,sentences):
        batch=[]
        token_embeddings=[]
        tokens_metadata={'freqs':[],'stopword_flag':[],'lowcase_flag':[],'subword_flag':[]}
        for sentence in tqdm(sentences):
            if not self.config['cased']:
                sentence=sentence.lower()
            batch.append(' '.join(sentence.replace('\n', '').split()[:self.config['max_length']]))
            if len(batch) >= BATCH_SIZE:
                self.process_batch(token_embeddings,tokens_metadata,batch)
                batch = []
        if len(batch) >= 0:
            self.process_batch(token_embeddings,tokens_metadata,batch)
        token_embeddings=list(zip(*token_embeddings))
        return token_embeddings, tokens_metadata

    def get_similarity_prompt(self,x,tokens_metadata):
        res_isot=0
        layer,x=x
        n=self.config['N']
        x=np.concatenate(x,axis=0)
        deterministic_shuffle(x)
        x=x[~(np.isnan(x).any(axis=1))]
        x=x[:n]
        n=x.shape[0]
        aux_metadata={}
        for k in tokens_metadata.keys():
            if len(tokens_metadata[k])>0:
                aux_metadata[k]=np.array(tokens_metadata[k])
                deterministic_shuffle(aux_metadata[k])
                aux_metadata[k]=aux_metadata[k][:n]
        tokens_metadata=aux_metadata
        x = torch.tensor(x)
        results_an=np.zeros((n,n))

        for i in tqdm(range(n)):   
            aux_sim=self.similarity.one_vs_all(x, i).detach().numpy()
            res_isot=res_isot+aux_sim.mean()
            results_an[i]=aux_sim

        if layer==12 and 'bias' in TASK:
            np.save('an.npy',results_an)
            for k in tokens_metadata.keys():
                 np.save(f'{k}.npy',tokens_metadata[k])
            sys.exit()

        return res_isot/n

    def read_stsb(self):
        sts_test=pd.read_csv('../stsbenchmark/sts-test_fixed.csv', header = None, sep='\t', quoting=csv.QUOTE_NONE, names=['genre', 'filename', 'year', 'score', 'sentence1', 'sentence2'])
        return sts_test['sentence1'].array ,sts_test['sentence2'].array ,sts_test['score'].array

    def sts_benchmark(self,name,combination_config):
        timer=Timer()
        self.model=self.get_model()
        timer()
        sents1,sents2,annots=self.read_stsb()
        timer()
        print('Computing embeddings(1)...')
        all_embeddings1=self.make_embeddings(sents1)[0]
        timer()
        print('Computing embeddings(2)...')
        all_embeddings2=self.make_embeddings(sents2)[0]
        timer()
        per_layer_sts=[]
        print('Computing benchmark...')
        for i in range(13):
            token_embeddings1=np.concatenate(all_embeddings1[i],axis=0)
            token_embeddings2=np.concatenate(all_embeddings2[i],axis=0)
            sims=self.similarity(torch.Tensor(token_embeddings1),torch.Tensor(token_embeddings2)).detach().numpy()
            not_nan_mask=~np.isnan(sims)
            sims=sims[not_nan_mask]
            per_layer_sts.append(spearmanr(sims,annots[not_nan_mask]).correlation)

        print(timer.get_times())
        print(per_layer_sts)

        rows=pd.DataFrame([dict(
            **combination_config,
            name=name,
            layer=i,
            STS=per_layer_sts[i],            
        ) for i in range(len(per_layer_sts))])
        return rows

    def isotropy(self,name,combination_config):
        timer=Timer()
        self.model=self.get_model()
        timer()
        sentences=self.read_wikisent()
        timer()
        print('Computing embeddings...')
        token_embeddings, tokens_metadata=self.make_embeddings(sentences)
        timer()
        print('Computing similarities...')
        per_layer_similarities=list(map(lambda x: self.get_similarity_prompt(x,tokens_metadata),tqdm(enumerate(token_embeddings))))
        timer()
        print(timer.get_times())
        print(per_layer_similarities)
        
        rows=pd.DataFrame([dict(
            **combination_config,
            name=name,
            layer=i,
            isotropy=per_layer_similarities[i],
            inference_time=timer.get_times()[0][2]/self.config['N'],
            
        ) for i in range(len(per_layer_similarities))])
        return rows

    def run_configuration(self):
        txt_name=self.config['dataset'].replace('.txt','').replace('.csv','')
        model_name=self.config['model'].split('/')[-1]
        name=f"{txt_name}|{model_name}|{self.config['pooling']}|{self.config['N']}|{self.config['max_length']}"
        combination_config={
            'dataset':txt_name,
            'model':model_name,
            'pooling':self.config['pooling'],
            'N':self.config['N'],
            'max_length':self.config['max_length'],
            'cased':self.config['cased'],
            'no_stop':self.config['no_stop'],
            'no_sub':self.config['no_sub'],
            'no_cls':self.config['no_cls'],
            'no_sep':self.config['no_sep']
        }

        query=[]
        for k,v in combination_config.items():
            if type(v)==str:
                query.append(f"{k} == '{v}'")
            else:
                query.append(f"{k} == {v}")

        query=' & '.join(query)

        if 'bias' in TASK or self.results.shape[0]==0 or self.results.query(query).shape[0]==0:
            if TASK=='STS':
                rows=self.sts_benchmark(name,combination_config)
            else:
                rows=self.isotropy(name,combination_config)
            self.results=self.results.append(rows)
            print(self.results)
            self.results.to_csv(OUT_PATH)
        return self.results

def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

def grid_search(configs):
    try:
        results=pd.read_csv(OUT_PATH,index_col=0)
    except:
        results=pd.DataFrame()
    configs=my_product(configs)
    print(pd.DataFrame(configs))
    for i in tqdm(range(len(configs))):
        config=configs[i]
        evaluator=Evaluator(config,results)
        results=evaluator.run_configuration()

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

    