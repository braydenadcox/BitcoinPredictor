import pandas as pd

df = pd.read_csv('distilbert_sentiment.csv')

df_score_only = df[['positive_score', 'negative_score']]  

df_score_only.to_csv('distilbert_scores_only.csv', index=False)