#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk pandas matplotlib wordcloud


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud, STOPWORDS

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

file_path = 'Apple-Twitter-Sentiment-DFE.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

print(data.head())
print(data.columns)

data = data[['text', 'sentiment', 'sentiment:confidence']]

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

data['clean_text'] = data['text'].apply(preprocess_text)

sid = SentimentIntensityAnalyzer()
data['vader_sentiment'] = data['clean_text'].apply(lambda x: sid.polarity_scores(x))

def categorize_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

data['vader_sentiment_category'] = data['vader_sentiment'].apply(lambda x: categorize_sentiment(x['compound']))

sentiment_counts = data['vader_sentiment_category'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Apple Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

def plot_wordcloud(sentiment):
    text = ' '.join(data[data['vader_sentiment_category'] == sentiment]['clean_text'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', max_words=200).generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment.capitalize()} Tweets')
    plt.show()

plot_wordcloud('positive')
plot_wordcloud('negative')
plot_wordcloud('neutral')


# In[ ]:




