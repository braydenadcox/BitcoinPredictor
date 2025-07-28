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
df.to_csv('data/roberta_sentiment.csv', index=False)