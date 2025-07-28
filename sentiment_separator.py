import pandas as pd

df = pd.read_csv('data/roberta_sentiment.csv')

single = df[['sentiment_score']]

single.to_csv('data/roberta_influence_score_only.csv', index=False)