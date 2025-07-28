from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('data/tweetSample.csv')

# Combine title and summary
df['combined'] = df['tweet_text'].fillna('')

# Lists to hold sentiment scores
positive_scores = []
negative_scores = []

# Load model/tokenizer once
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Sets up integration for GPU if the model is powerful enough (I have an RTX 5080 so it should be okay)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Batch size and model configuration, can easily go up from the base layer of 128
batch_size = 128

# =============================================================================
# DATAFRAME PREPARATION AND PROCESSING BEGINS HERE
# =============================================================================

for i in range(0, len(df), batch_size):
    batch = df['combined'][i:i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(batch.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Run model through the batch
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Grab positive and negative scores
    negative_probs = probs[:, 0].cpu().numpy()
    positive_probs = probs[:, 1].cpu().numpy()

    # Add the batch results to the lists
    positive_scores.extend(positive_probs * 100)
    negative_scores.extend(negative_probs * 100)

# Assign scores to dataframe
df['positive_score'] = positive_scores
df['negative_score'] = negative_scores
df['sentiment_score'] = (df['positive_score'] - df['negative_score']) / 100

# Save to CSV
df.to_csv('data/distilbert_finetuned_sentiment.csv', index=False)


'''# Testing output and results
print(df[['username', 'followers_count', 'influence_raw', 'influence_scaled', 'positive_score', 'weighted_positive']].sort_values('followers_count', ascending=False).head(10))
print(df['influence_scaled'].describe())'''
