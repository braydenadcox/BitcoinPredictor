from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

# Set up RoBERTa tokenizer and model variables
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model.eval()

df = pd.read_csv('data/tweetSample.csv')

# GPU integration and batch size configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_size = 128

# Lists to hold sentiment scores
sentiment_score = []

# =============================================================================
# IMPLEMENTATION OF INFLUENCE FUNCTION FOR SENTIMENT ANALYSIS
# =============================================================================

for col in ['like_count', 'reply_count', 'retweet_count', 'quote_count', 'followers_count']:
    df[col] = df[col].fillna(0)

wl, wr, wrt, wq, wf = 0.3, 0.1, 0.5, 0.3, 2.0

df['influence_raw'] = (
    wl * np.log1p(df['like_count']) +
    wr * np.log1p(df['reply_count']) +
    wrt * np.log1p(df['retweet_count']) +
    wq * np.log1p(df['quote_count']) +
    wf * np.log1p(df['followers_count'])
)

influence_max = df['influence_raw'].max() or 1.0
df['influence_scaled'] = 1.0 + (df['influence_raw'] / influence_max)

# =============================================================================
# DATAFRAME PREPARATION AND PROCESSING BEGINS HERE
# =============================================================================

for i in range(0, len(df), batch_size):
    batch = df['tweet_text'][i:i + batch_size].fillna('')

    # Tokenize the batch
    inputs = tokenizer(batch.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    # Run model through the batch
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Grab sentiment scores
    batch_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy()
    sentiment_score.extend(batch_scores)

df['sentiment_score'] = sentiment_score
df['weighted_sentiment'] = df['sentiment_score'] * df['influence_scaled']

# Filter out irrelevant columns (Only need timestamp and sentiment scores)
df = df[['create_at', 'sentiment_score' 'weighted_sentiment']].copy()
df.columns = ['timestamp', 'sentiment_score', 'weighed_sentiment']

# Optimization of sentiment score values
df['sentiment_score'] = df['sentiment_score'].round(5)
df['weighed_sentiment'] = df['weighed_sentiment'].round(5)

df.to_csv('data/roberta_weighed_sentiment.csv', index=False)