import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
import torch
from torch_geometric.data import HeteroData
from datetime import timedelta
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
from day_graphs import DayGraphs_Creation
import json

with open("../config_files/bloomberg_creation_config.json", 'r') as f:
    config = json.load(f) 

dfs = []
for index in ["nyse", "nasdaq", "nysemkt"]:
  for i in os.listdir(f"{config['stocks_path']}/data/daily/us/{index} stocks/"):
    if i in ["1","2","3","4","5"]:
      for k in os.listdir(f"{config['stocks_path']}/data/daily/us/{index} stocks/{i}"):
        #symbol	dateOfPrice	open	high	low	close	volume
        dfs.append(pd.read_csv(f"{config['stocks_path']}/data/daily/us/{index} stocks/{i}/{k}", sep=",", header=0, names = ["symbol","<PER>","dateOfPrice","<TIME>","open","high","low","close","volume","<OPENINT>"]))
    else:
        dfs.append(pd.read_csv(f"{config['stocks_path']}/data/daily/us/{index} stocks/{i}", sep=",", header=0, names = ["symbol","<PER>","dateOfPrice","<TIME>","open","high","low","close","volume","<OPENINT>"]))


all_stocks = pd.concat(dfs)
del dfs
all_stocks.columns =  ["symbol","<PER>","dateOfPrice","<TIME>","open","high","low","close","volume","<OPENINT>"]

"""
date was in the format of
20190101
day is first 2 digits month is the 2nd 2 digits
"""
import datetime
def to_datetime(x:int):
  day = x%100
  x  = x//100
  month = x%100
  year = x//100
  return datetime.datetime(year,month,day)


all_stocks["dateOfPrice"] = all_stocks["dateOfPrice"].apply(to_datetime)
analysis_df = all_stocks[(all_stocks["dateOfPrice"] <= datetime.datetime(2013, 11, 26)) & (all_stocks["dateOfPrice"]>= datetime.datetime(2006, 10, 20))].sort_values("dateOfPrice").reset_index(drop=True).copy()
del all_stocks

######################  the next files can downloaded using
# import gdown
# gdown.download_folder("https://drive.google.com/drive/folders/1aNyfr_BX7O2G1QTHcBIFhO2jGjm5CbeY?usp=drive_link")
###################### 
company_sentences = pd.read_csv(f"{config['bloomberg_path']}/company_sentences.csv")
company_interaction = pd.read_csv(f"{config['bloomberg_path']}/company_interaction.csv")
sub_tickers = list(set(list(company_interaction["ticker1"].unique()) + list(company_interaction["ticker2"].unique())))
company_sentences["tickers"] = company_sentences["tickers"].apply(lambda x: x[1:-1].replace("'","").replace(" ", "").split(","))

analysis_df["Ticker"] = analysis_df["symbol"].apply(lambda x: x[:-3])
analysis_df = analysis_df[analysis_df["Ticker"].isin(sub_tickers)]


import numpy as np
use_std_label = True
#dict of a stock history df for each company
analysis_df = analysis_df.sort_values("dateOfPrice").reset_index(drop=True)
company_stocks = {}
for symbol, data in analysis_df.groupby('Ticker')[analysis_df.columns]: #groupby company
  # checks every close of the day if it's higher that the close in the day before and creates a binary array
  # data["out"] is the target
  # break
  if use_std_label:
    data["target"] = np.where(data["close"] - data["close"].shift(1).fillna(0)> 0.1*np.std(data["close"].tolist()[:50]), 1, 0 )
  else:
    data["target"] = np.where(data["close"] - data["close"].shift(1).fillna(0)> 0, 1, 0 )
  data.index = pd.to_datetime(data.dateOfPrice) # setting the index as the dateOfPrice for faster look ups when creating target output for each week_graph
  company_stocks[symbol] = data



tokenizer = AutoTokenizer.from_pretrained("Sigma/financial-sentiment-analysis")

sigma =  AutoModel.from_pretrained("Sigma/financial-sentiment-analysis")

sigma.eval()

if torch.cuda.is_available():
  sigma = sigma.to("cuda")

device = next(sigma.parameters()).device

text =["this is a test"] * 10

def get_article_vectors(text:List[str], batch_size=32, max_length=100)-> torch.Tensor:
  all_vectors = []
  with torch.no_grad():
    with torch.autocast(device_type="cuda"): # faster inference
      for i in tqdm(range(0,len(text),batch_size)):
        # tokenizer is used to convert news strings into token_ids
        text_tokens = tokenizer(text[i:i+batch_size],padding=True, truncation= True, max_length=max_length, return_tensors='pt')
        for k,v in text_tokens.items():
          text_tokens[k] = v.to(device)
        out = sigma(**text_tokens,)
        pooler_output = out.pooler_output
        all_vectors.append(pooler_output.cpu().detach())

  all_vectors = torch.cat(all_vectors, axis=0 )
  return all_vectors #



company_sentences["datetime"] = pd.to_datetime(company_sentences["datetime"])
company_sentences = company_sentences.sort_values("datetime").reset_index(drop=True)
sentence_vectors = get_article_vectors(company_sentences["sentence"].tolist()) # I thought we didn't need to save the vectors as they only take 4 minutes to produce



# edge_types
# article - main_company - company
# article - mentioned - company



lag = 21
stock_days = lag - lag// 7 * 2 # saturday and friday for each week

company_sentences["datetime"] = pd.to_datetime(company_sentences["datetime"])
company_interaction["datetime"] = pd.to_datetime(company_interaction["datetime"])
company_to_index = {k :v for v,k in zip(range(len(company_stocks.keys())), company_stocks.keys()) }
index_to_company = {v:k for k,v in company_to_index.items()} #reverse
week_graphs = []
split_index = 0
for day in pd.to_datetime(pd.Series(analysis_df["dateOfPrice"].unique()[lag:])):
  start = day - timedelta(lag)
  print(day,  start)

  target_news = company_sentences[(company_sentences["datetime"]>= start) & (company_sentences["datetime"] < day) ].copy()
  target_interaction = company_interaction[(company_interaction["datetime"]>= start) & (company_interaction["datetime"] < day) ].copy()
  week_graph = HeteroData()
  edges = {}
  # creating edge_index
  edges["sentence-mentioned-company"] = [[],[]]
  edges["company-mentioned_with-company"] = [[],[]]
  edges["company-mentioned_in-sentence"] = [[],[]]

  # creating an array that says stock price info(gone up or down) exists for that day and company or not
  # if info doesn't exist we won't consider the predictions for that day and company
  info_exists = [ day in  company_stocks[index_to_company[i]].index for i in range(len(company_to_index.keys()))]

  y = [ company_stocks[index_to_company[i]].loc[day]["target"] if info_exists[i] else 0  for i in range(len(company_to_index.keys())) ]

  company_timeseries =[ company_stocks[index_to_company[i]].loc[start:day-timedelta(1)][["open", "high", "low", "close", "volume"]].to_numpy()  for i in range(len(company_to_index.keys())) ]
  # break
  missing_prices =[ False if x.shape[0] == stock_days else True  for x in company_timeseries  ]
  # break
  company_timeseries =[ x if len(x) == stock_days else np.concatenate([x, np.nan_to_num(x.mean(axis=0).reshape(1,-1), nan=0)+np.zeros((int(stock_days-len(x)), x.shape[1]))])   for x in company_timeseries]
  company_timeseries = np.concatenate([np.expand_dims(x, 0) for x in company_timeseries])
  # # creating gaph edge_index
  for i, (_, r) in enumerate(target_news.iterrows()):
    for comp in r["tickers"]:
      if comp not in company_to_index.keys(): continue
      edges["sentence-mentioned-company"][0].append(i)
      edges["sentence-mentioned-company"][1].append(company_to_index[comp])
      edges["company-mentioned_in-sentence"][1].append(i)
      edges["company-mentioned_in-sentence"][0].append(company_to_index[comp])

  for i, (_, r) in enumerate(target_interaction.iterrows()):
    if r["ticker1"] in company_to_index.keys() and r["ticker2"] in company_to_index.keys():
      edges["company-mentioned_with-company"][0].append(company_to_index[r["ticker1"]])
      edges["company-mentioned_with-company"][1].append(company_to_index[r["ticker2"]])

      edges["company-mentioned_with-company"][1].append(company_to_index[r["ticker1"]])
      edges["company-mentioned_with-company"][0].append(company_to_index[r["ticker2"]])


  for k, v in edges.items():
    edge_name = k.split("-")
    week_graph[edge_name[0],edge_name[1],edge_name[2]].edge_index  = torch.tensor(v)

  week_graph["target"] = torch.as_tensor(y)
  week_graph["missing_prices"] = torch.as_tensor(missing_prices)
  week_graph["info_exists"] = torch.as_tensor(info_exists)
  week_graph["company_timeseries"] =torch.from_numpy(company_timeseries)
  week_graph["company"].x = torch.as_tensor(range(len(index_to_company.keys()))) # just the ids to be passed to the embedding layer
  week_graph["sentence"].x =  sentence_vectors[target_news.index]  # get_article_vectors(target_news["content"].to_list()) # Nx768 (number of articles X the embedding dim of finbert)
  week_graph["date"] = day
  week_graphs.append(week_graph)
  if len(week_graphs) >= 500:
    DayGraphs_Creation(f"{config['bloomberg_path']}/bloomberg_graph_{split_index}", week_graphs)
    week_graphs = []
    split_index +=1
if len(week_graphs) > 0 :
  DayGraphs_Creation(f"{config['bloomberg_path']}/bloomberg_graph_{split_index}", week_graphs)


# 25 24 23 22 21 20