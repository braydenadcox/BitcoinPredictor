import pandas as pd

df = pd.read_csv('data/distilbert_finetuned_sentiment.csv')

single = df[['sentiment_score']]

single.to_csv('data/distilbert_finetuned_influence_score_only.csv', index=False)