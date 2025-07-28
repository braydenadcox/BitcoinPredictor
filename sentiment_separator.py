import pandas as pd

df = pd.read_csv('data/roberta_weighed_sentiment.csv')

single = df[['weighted_sentiment']]

single.to_csv('data/roberta_weighed_score_only.csv', index=False)