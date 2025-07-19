import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def FinBERT_sentiment_score(heading):
    """
    compute sentiment score using pretrained FinBERT on -1 to 1 scale. -1 being negative and 1 being positive
    """
    result = nlp(heading)
    if result[0]['label'] == "positive":
        return result[0]['score']
    elif result[0]['label'] == "neutral":
        return 0
    else:
        return (0 - result[0]['score'])


def VADER_sentiment_score(heading):
    """
    compute sentiment score using pretrained VADER on -1 to 1 scale. -1 being negative and 1 being positive
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    # from nltk.sentiment.vader import SentimentIntensityAnalyzer -- second import statement is unnecessary
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(heading)
    if result['pos'] == max(result['neg'], result['neu'], result['pos']):
        return result['pos']
    if result['neg'] == max(result['neg'], result['neu'], result['pos']):
        return (0 - result['neg'])
    else:
        return 0

tweet_df = pd.read_csv("tweetSample.csv")



BERT_sentiment = []


for i in tqdm(range(len(tweet_df)), desc="Processing headlines"):
    news_list = tweet_df.iloc[i, 1:].tolist()
    news_list = [i for i in news_list if i != '0']
    text = " ".join([str(x) for x in news_list if pd.notna(x)])
    score_BERT = FinBERT_sentiment_score(text)
    BERT_sentiment.append(score_BERT)


# print(tweet_df.iloc[129])

tweet_df['FinBERT score'] = BERT_sentiment

tweet_df.to_csv("finbertTweetSentiment.csv")
# tweet_df[['title', 'summary', 'published_date', 'FinBERT score']].to_csv('finbert_train_test.csv', index=False)
