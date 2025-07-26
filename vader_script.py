import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Gets the News articles
print("Vader is running")
df = pd.read_csv('../../../news.csv')

#combines the title and summary
df['combined'] = df['title'].fillna('') + ': ' + df['summary'].fillna('')
# Prepare lists to store results
overall_list = []
positive_scores = []
neutral_scores = []
negative_scores = []
# loops through the news articals
for val in df['combined']:
# sets up vader and uses vader
    sid_obj = SentimentIntensityAnalyzer()
    dict = sid_obj.polarity_scores(val)

    # 1 is positive -1 is negative, 0 is neutral
    if( dict['compound'] > 0.05 ):
        overall = 1
    elif ( dict['compound'] < -0.05 ):
        overall = -1
    else:
        overall = 0
    #saves all the neg and pos scores

    overall_list.append(overall)
    positive_scores.append(dict['pos'] * 100)
    neutral_scores.append(dict['neu'] * 100)
    negative_scores.append(dict['neg'] * 100)

# Add the sentiment results to the DataFrame
df['overall'] = overall_list
df['positive_score'] = positive_scores
df['neutral_score'] = neutral_scores
df['negative_score'] = negative_scores

# Save to CSV
df.to_csv('vader_news.csv', index=False)