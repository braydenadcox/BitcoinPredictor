from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

print("bad " + "good")

# Load CSV
df = pd.read_csv('../../../news.csv')
print("bad " + "good")

# Combine title and summary
df['combined'] = df['title'].fillna('') + ': ' + df['summary'].fillna('')

# Lists to hold sentiment scores
positive_scores = []
negative_scores = []

# Load model/tokenizer once
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Loop through combined text
for value in df['combined']:
    print(value)
    if value:
        text = value
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        negative_prob = probs[0][0].item()
        positive_prob = probs[0][1].item()

        print(f"Negative: {negative_prob:.4f}")
        print(f"Positive: {positive_prob:.4f}")

        positive_scores.append(round(positive_prob * 100, 3))
        negative_scores.append(round(negative_prob * 100, 3))
    else:
        positive_scores.append(None)
        negative_scores.append(None)

# Assign scores to dataframe
df['positive_score'] = positive_scores
df['negative_score'] = negative_scores

# Save to CSV
df.to_csv('news_sentiment.csv', index=False)