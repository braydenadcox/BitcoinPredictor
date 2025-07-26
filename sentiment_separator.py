import pandas as pd

df = pd.read_csv('data/vader_news.csv')

df['combined'] = df['positive_score'] - df['negative_score']

df.to_csv('distilbert_scores_only.csv', index=False)