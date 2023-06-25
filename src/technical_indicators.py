import numpy as np
import scipy.sparse as sp
import pandas as pd 

def rsi_indicator(stock_history:pd.DataFrame, rsi_period:int):
  stock_history.sort_values(by=['symbol', 'dateOfPrice'], inplace=True)
  companies = stock_history["symbol"].unique()
  for company in companies:
      # Subset the stock history data for the current company
      company_history = stock_history[stock_history["symbol"] == company]
      company_history.sort_values(by=['dateOfPrice'], inplace=True)
      # Calculate RSI for the current company
      delta = company_history["close"].diff()
      gain = delta.where(delta > 0, 0)
      loss = -delta.where(delta < 0, 0)
      avg_gain = gain.rolling(window=rsi_period).mean()
      avg_loss = loss.rolling(window=rsi_period).mean()
      rs = avg_gain / avg_loss
      company_history["rsi"] = 100 - (100 / (1 + rs))
      company_mean = company_history["rsi"].mean()
      company_history = company_history.fillna(company_mean)
      # Add the rsi data to the dictionary
      stock_history.loc[stock_history["symbol"] == company, "rsi"] = company_history["rsi"].values
  return stock_history

def simple_moving_average_indicator(stock_history:pd.DataFrame, window:int):
  companies = stock_history["symbol"].unique()
  stock_history.sort_values(by=['symbol', 'dateOfPrice'], inplace=True)
  # stock_history= stock_history.sort_values(by=['symbol', 'dateOfPrice'])
  for company in companies:
      # Subset the stock history data for the current company
      company_history = stock_history[stock_history["symbol"] == company]
      company_history.sort_values(by=['dateOfPrice'], inplace=True)
      company_history[['smv_open', 'smv_high', 'smv_low', 'smv_close', 'smv_volume']] = company_history[['open', 'high', 'low', 'close', 'volume']].rolling(window).mean() 
      company_mean = company_history[['smv_open', 'smv_high', 'smv_low', 'smv_close', 'smv_volume']].mean()
      #print()
      company_history= company_history[['smv_open', 'smv_high', 'smv_low', 'smv_close', 'smv_volume']].fillna(company_mean)
      stock_history.loc[stock_history["symbol"] == company, ['smv_open', 'smv_high', 'smv_low', 'smv_close', 'smv_volume']] = company_history[['smv_open', 'smv_high', 'smv_low', 'smv_close', 'smv_volume']].values
  return stock_history
      

# stock_df = simple_moving_average_indicator(stock_df, 4)
# stock_df
# stock_df = pd.read_csv("/content/drive/MyDrive/Stock Market Prediction Graduation/processed_stock_df.csv")
# stock_df.sort_values(by=['symbol', 'dateOfPrice'], inplace=True)
# stock_df = RSI_indicator(stock_df, 4)
# stock_df
