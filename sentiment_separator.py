import pandas as pd

df = pd.read_csv('data/distilbert_finetuned_influence_sentiment.csv')

single = df[['weighted_positive', 'weighted_negative']]

single.to_csv('data/distilbert_finetuned_influence_score_only.csv', index=False)