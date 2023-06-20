import torch
from tqdm import tqdm
from typing import List


def get_companies(article:str, ner)-> List[str]:
    doc = ner(article)
    results = []
    for ent in doc.ents:
        results.append(ent.label_)
    return list((set(results)))

def get_article_vectors(text:List[str], sigma, tokenizer, device, batch_size=32, max_length=100)-> torch.Tensor:
    all_vectors = []
    with torch.no_grad():
        for i in tqdm(range(0,len(text),batch_size)):
            # tokenizer is used to convert news strings into token_ids
            text_tokens = tokenizer(text[i:i+batch_size],padding=True, truncation= True, max_length=max_length, return_tensors='pt')
            for k,v in text_tokens.items():
                text_tokens[k] = v.to(device)

            out = sigma(**text_tokens, )
            pooler_output = out.pooler_output
            all_vectors.append(pooler_output.cpu().detach())

    all_vectors = torch.cat(all_vectors, axis=0 )
    return all_vectors