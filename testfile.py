import yfinance as yf
import pandas as pd

bitcoin = yf.Ticker("BTC-USD")
bitcoin_data = bitcoin.history(period="1d", interval="1m")
print(bitcoin_data)