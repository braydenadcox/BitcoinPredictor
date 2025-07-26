import pandas as pd

df = pd.read_csv('data/distilbert_finetuned_sentiment.csv')

single = df[['positive_score', 'negative_score']]

single.to_csv('data/distilbert_finetuned_score_only.csv', index=False)