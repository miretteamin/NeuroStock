import torch
from tqdm import tqdm
from typing import List
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import DataLoader
def get_companies(article:str, ner)-> List[str]:
    doc = ner(article)
    results = []
    for ent in doc.ents:
        results.append(ent.label_)
    return list((set(results)))

#Get news node embedding 
def get_article_vectors(text:List[str], model, tokenizer, device, type, batch_size=32, max_length=100)-> torch.Tensor:
    all_vectors = []
    with torch.no_grad():
        for i in tqdm(range(0,len(text),batch_size)):
            # tokenizer is used to convert news strings into token_ids
            text_tokens = tokenizer(text[i:i+batch_size],padding=True, truncation= True, max_length=max_length, return_tensors='pt')
            for k,v in text_tokens.items():
                text_tokens[k] = v.to(device)

            out = model(**text_tokens, )

            if type == "sigma":
                pooler_output = out.pooler_output    
                all_vectors.append(pooler_output.cpu().detach())

            elif type == "finbert":
                last_hidden_state = out.last_hidden_state
                # print(np.shape(last_hidden_state))
                cls_hidden_state = last_hidden_state[:, 0]
                all_vectors.append(cls_hidden_state.cpu().detach())

    all_vectors = torch.cat(all_vectors, axis=0 )
    return all_vectors

def read_config_file(config_file):
    with open(config_file, "r") as f:
        data = json.load(f)
    return data

def news_emb_model(type="finbert"):
    if type == "finbert":
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModel.from_pretrained("ProsusAI/finbert")
        return model, tokenizer

    elif type == "sigma":
        tokenizer = AutoTokenizer.from_pretrained("Sigma/financial-sentiment-analysis")
        model = AutoModel.from_pretrained("Sigma/financial-sentiment-analysis")
        return model, tokenizer
    


def split_train_test(graphs, train_percentage=0.7, batch_size=8):
    
    train = graphs[:int(train_percentage*len(graphs))]
    test = graphs[int(train_percentage*len(graphs)):]
    print(len(train),len(test))

    # return Train Dataloader, Test Dataloader
    return DataLoader(train, batch_size=batch_size), DataLoader(test, batch_size=batch_size, shuffle=False)
