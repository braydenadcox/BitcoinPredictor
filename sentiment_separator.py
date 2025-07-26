import pandas as pd

df = pd.read_csv('data/vader_news.csv')

single = df[['positive_score', 'negative_score', 'neutral_score']]

single.to_csv('data/vader_score_only.csv', index=False)