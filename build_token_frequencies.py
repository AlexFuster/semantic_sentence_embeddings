from transformers import AutoTokenizer
from tqdm import tqdm
import json

BATCH_SIZE=128
max_length=32
N=100000
model_name="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
token_frequencies={}


def read_wikisent(dataset,n_lines=10):
    with open(f'../{dataset}','r') as f:
        sentences=[]
        for i in range(n_lines):
            sentences.append(f.readline())
    return sentences

def update_freqs(batch):
    batch = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True)

    for x in batch.input_ids:
        for token in tokenizer.convert_ids_to_tokens(x):
            token_frequencies[token]=token_frequencies.get(token,0)+1

sentences=read_wikisent('wiki1m_for_simcse_shuf.txt',N)
batch=[]
for sentence in tqdm(sentences):
    batch.append(' '.join(sentence.replace('\n', '').split()))
    if len(batch) >= BATCH_SIZE:
        update_freqs(batch)
        batch=[]

if len(batch) >= 0:
    update_freqs(batch)

token_frequencies=dict(sorted(token_frequencies.items(),key=lambda x:x[1],reverse=True))

with open('token_freqs.json','w') as f:
    json.dump(token_frequencies,f)
