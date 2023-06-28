from torch_geometric.data import HeteroData
from datetime import timedelta
import torch
from tqdm.notebook import tqdm
from torch_geometric.data import HeteroData
"""
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
from torch_geometric.data  import InMemoryDataset

import numpy as np
import pandas as pd

from helper_fns import get_article_vectors, news_emb_model


class DayGraphsCreation(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        #print(len(self.data_list))
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #self.process()

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


class DayGraphs(InMemoryDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'
    

def save_news_vectors(config):
    news_df = pd.read_csv(config["data"]["processed_news_df_path"])
    news_df["mentioned_companies"] = news_df["mentioned_companies"].apply(lambda x: str(x)[1:-1].replace("'","").replace(" ", "").split(","))
    stock_df = pd.read_csv(config["data"]["processed_stock_df_path"])

    # Dict of a stock history df for each company
    company_stocks = {}
    for symbol, data in stock_df.groupby('symbol')[stock_df.columns]: #groupby company
        # checks every close of the day if it's higher that the close in the day before and creates a binary array
        # data["out"] is the target
        data["target"] = np.where(data["close"] - data["close"].shift(1).fillna(0) > 0, 1, 0)
        data.index = pd.to_datetime(data.dateOfPrice) # setting the index as the dateOfPrice for faster look ups when creating target output for each week_graph
        company_stocks[symbol] = data

    # Each Symbol has its own historical data


    # Target length is 1296
    data["target"].value_counts()[1] # 1 --> 690  & 0 --> 606

    import socket
    from urllib3.connection import HTTPConnection

    HTTPConnection.default_socket_options = ( 
        HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000), #1MB in byte
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
    ])

    model, tokenizer = news_emb_model(type=config["graphs"]["news_emb_model"])

    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    device = next(model.parameters()).device

    news_vectors = get_article_vectors(list(news_df[config["graphs"]["news_part"]]), model, tokenizer, device, type=config["graphs"]["news_emb_model"])

    # Save news vectors
    torch.save(news_vectors, config["graphs"]["news_vectors_path"])
    return company_stocks


def save_graphs(config, company_stocks):
    # edge_types
    # article - main_company - company
    # article - mentioned - company
    # article - in_industry - industry
    # company - in_industry - industry


    lag = 21
    stock_days = lag - lag// 7 * 2 # saturday and friday for each week

    nasdaq_screener = pd.read_csv(config["data"]["nasdaq_screener_path"])
    news_df = pd.read_csv(config["data"]["processed_news_df_path"])
    stock_df = pd.read_csv(config["data"]["processed_stock_df_path"])
    news_vectors = torch.load(config["graphs"]["news_vectors_path"])

    news_df["release_date"] = pd.to_datetime(news_df["release_date"])
    company_to_industry = nasdaq_screener.groupby("Symbol")["Sector"].agg(lambda x: list(x)[0]) # agg list(x)[0] as x is just a list of repeated sectors and we need 1 sector for each company
    # company_to_industry["ABMD"] = "Health Care"
    industry_to_index = {k : v for v, k in enumerate(nasdaq_screener["Sector"].unique()) }
    company_to_index = {k : v for v, k in zip(range(news_df["symbol"].nunique()), news_df["symbol"].unique()) }
    index_to_company = {v : k for k, v in company_to_index.items()} #reverse
    day_graphs = [] 

    for day in pd.to_datetime(pd.Series(stock_df["dateOfPrice"].unique()[lag:])):

        start = day - timedelta(lag)
        # print(day,  start)

        target_news = news_df[(news_df["release_date"]>= start) & (news_df["release_date"] < day) ].copy()

        day_graph = HeteroData()
        edges = {}
        # creating edge_index
        edges["article-main_company-company"] = [[],[]]
        edges["article-mentioned-company"] = [[],[]]
        edges["article-in_industry-industry"] = [[],[]]
        edges["company-mentioned_in-article"] = [[],[]]
        edges["company-in_industry-industry"] = [[],[]]
        edges["industry-has_company-company"] = [[],[]]

        for company in company_to_index.keys():
            edges["company-in_industry-industry"][0].append(company_to_index[company])
            edges["company-in_industry-industry"][1].append(industry_to_index[company_to_industry[company]])
            edges["industry-has_company-company"][0].append(industry_to_index[company_to_industry[company]])
            edges["industry-has_company-company"][1].append(company_to_index[company])
            # print(company)
            # print(company_to_index[company])
            # print(company_to_industry[company])
            # print(edges["company-in_industry-industry"])
            # print(edges["industry-has_company-company"])

        
        # creating an array that says stock price info(gone up or down) exists for that day and company or not
        # if info doesn't exist we won't consider the predictions for that day and company
        info_exists = [ day in company_stocks[index_to_company[i]].index for i in range(news_df["symbol"].nunique() )]

        y = [ company_stocks[index_to_company[i]].loc[day]["target"] if info_exists[i] else 0  for i in range(news_df["symbol"].nunique()) ]

        company_timeseries = [company_stocks[index_to_company[i]].loc[start:day-timedelta(1)][["open", "high", "low", "close", "volume"]].to_numpy()  for i in range(news_df["symbol"].nunique()) ]

        ## Check for missing prices
        missing_prices = [ False if x.shape[0] == stock_days else True  for x in company_timeseries ]

        company_timeseries =[ x if len(x) == stock_days else np.concatenate([x, np.nan_to_num(x.mean(axis=0).reshape(1,-1), nan=0)+np.zeros((int(stock_days-len(x)), x.shape[1]))])   for x in company_timeseries]
        company_timeseries = np.concatenate([np.expand_dims(x, 0) for x in company_timeseries])

        # # creating gaph edge_index
        # print(target_news)
        for i, (_, r) in enumerate(target_news.iterrows()):
            # print(i)
            edges["article-main_company-company"][0].append(i)
            edges["article-main_company-company"][1].append(company_to_index[r["symbol"]])
        
            for comp in r["mentioned_companies"]:
                if comp not in company_to_index.keys(): continue
                edges["article-mentioned-company"][0].append(i)
                edges["article-mentioned-company"][1].append(company_to_index[comp])
                edges["company-mentioned_in-article"][1].append(i)
                edges["company-mentioned_in-article"][0].append(company_to_index[comp])

            edges["article-in_industry-industry"][0].append(i)
            edges["article-in_industry-industry"][1].append(industry_to_index[company_to_industry[r["symbol"]]])

        for k, v in edges.items():
            edge_name = k.split("-")
            day_graph[edge_name[0],edge_name[1],edge_name[2]].edge_index  = torch.tensor(v)

        day_graph["target"] = torch.as_tensor(y)
        day_graph["missing_prices"] = torch.as_tensor(missing_prices)
        day_graph["info_exists"] = torch.as_tensor(info_exists)
        day_graph["company_timeseries"] =torch.from_numpy(company_timeseries)
        day_graph["company"].x = torch.as_tensor(range(len(index_to_company.keys()))) # just the ids to be passed to the embedding layer
        day_graph["article"].x =  news_vectors[target_news.index]  # get_article_vectors(target_news["content"].to_list()) # Nx768 (number of articles X the embedding dim of finbert)
        day_graph["industry"].x = torch.as_tensor(range(len(industry_to_index.values())))
        day_graph["date"] = day
        day_graphs.append(day_graph)

        # 25 24 23 22 21 20
    DayGraphsCreation(config["graphs"]["graph_path"], day_graphs)
    return day_graphs