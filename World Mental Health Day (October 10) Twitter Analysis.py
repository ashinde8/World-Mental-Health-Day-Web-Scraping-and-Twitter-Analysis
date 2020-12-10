#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
import seaborn as sns
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re
from nltk.stem import WordNetLemmatizer


# # World Mental Health Twitter Sentiment Analysis

# In[3]:


from IPython.display import Image
Image(filename='world-mental-health-day.png')


# In[7]:


pip install vaderSentiment


# In[3]:


mental_health = pd.read_csv('world_mental_health_day.csv')


# In[4]:


mental_health.drop(['COMMENTS','LIKES'], axis=1,inplace=True)


# In[5]:


mental_health.columns = ['Tweets', 'Retweets']


# In[6]:


mental_health.isnull().sum()


# In[7]:


mental_health.fillna(0, inplace=True)


# In[8]:


mental_health.isnull().sum()


# In[9]:


mental_health["word_count"] = mental_health["Tweets"].apply(lambda tweet: len(tweet.split()))


# In[10]:


wmh = pd.read_csv('WMHDayTweets.csv')


# In[11]:


wmh.drop('Hashtags', axis=1,inplace=True)


# In[12]:


mental_health.head()


# In[13]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[14]:


senti_analyzer = SentimentIntensityAnalyzer()


# In[15]:


compound_score = []

for sen in mental_health['Tweets']:
    
    compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])


# In[16]:


mental_health['Compound Score'] = compound_score


# In[17]:


Sentiment = []

for i in compound_score:
    
    if i >= 0.05:
        
        Sentiment.append('Positive')
        
    elif i > -0.05 and i < 0.05:
        
        Sentiment.append('Neutral')
        
    else:
        
        Sentiment.append('Negative')
        


# In[18]:


mental_health['Sentiment'] = Sentiment


# In[19]:


mental_health.head()


# In[20]:


wmh.head()


# In[21]:


mental_health.isnull().sum()


# In[22]:


mental_health_data = pd.concat([mental_health, wmh])


# # Tweets Dataset

# In[23]:


mental_health_data.head()


# In[24]:


mental_health_data.shape


# In[25]:


mental_health_data.isnull().sum()


# In[26]:


dfs = mental_health_data


# # Tweets Word Count Distribution

# In[27]:


# Word Count Distribution Histogram
sns_plot = sns.distplot(dfs['word_count'])
fig = sns_plot.get_figure()
fig.savefig("wmh_word_count.png")
sns.distplot(dfs['word_count'])


# In[28]:


pos_count = sum(dfs['Sentiment']=='Positive')
neg_count = sum(dfs['Sentiment']=='Negative')
neu_count = sum(dfs['Sentiment']=='Neutral')


# # Sentiment Distribution of the Tweets

# In[29]:


# Sentiment Distribution

pos_count = sum(dfs['Sentiment']=='Positive')
neg_count = sum(dfs['Sentiment']=='Negative')
neu_count = sum(dfs['Sentiment']=='Neutral')

import matplotlib.pyplot as plt
# Pie chart
labels = ['Positive tweets', 'Negative tweets', 'Neutral reviews']
sizes = [pos_count, neg_count, neu_count]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05)
 
    
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.savefig('wmh_sentiment_distribution.png')

plt.show()


# # Most Positive Tweet

# In[30]:


# Most Positive Tweet

pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]
pos_max


# # Most Negative Tweet

# In[31]:


# Most Negative Tweet

neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]
neg_max


# # Positive Tweets

# In[32]:


# Positive Tweets

gp = dfs.groupby(by=['Sentiment'])
positive_tweets = gp.get_group('Positive')
positive_tweets.head()


# In[33]:


positive_tweets.shape


# # Negative Tweets

# In[34]:


# Negative Tweets

negative_tweets = gp.get_group('Negative')
negative_tweets.head()


# In[35]:


negative_tweets.shape


# # Neutral Tweets

# In[36]:


# Neutral Tweets

neutral_tweets = gp.get_group('Neutral')
neutral_tweets.head()


# In[37]:


neutral_tweets.shape


# # Wordcloud Function

# In[38]:


# Wordcloud Function

def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# In[39]:


stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['worldmentalhealthday','worldmentalheathday']
stopwords.extend(newStopWords)


# In[40]:


def wordcloud(data):
    
    words_corpus = ''
    words_list = []

    
    for rev in data["Tweets"]:
        
        text = str(rev).lower()
        text = text.replace('rt', ' ') 
        
        
        text = re.sub(r"http\S+", "", text)        
        text = re.sub(r'[^\w\s]','',text)
        text = ''.join([i for i in text if not i.isdigit()])
        
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords]
        
        for word in tokens[:]: 
            if word.startswith('@'): 
                tokens.remove(word) 
        # Remove aplha numeric characters
        
        for words in tokens:
            
            words_corpus = words_corpus + words + " "
            words_list.append(words)
            
    return words_corpus, words_list


# In[43]:


import cv2
import numpy as np
image1 = cv2.imread('mask1.png')
mask = np.array(image1)


# # WordCloud - Positive Tweets

# In[44]:


# WordCloud - Positive Tweets

from wordcloud import WordCloud
positive_wordcloud = WordCloud(background_color= "white",mask=mask,width=900, height=500).generate(wordcloud(positive_tweets)[0])
    
plot_Cloud(positive_wordcloud)
positive_wordcloud.to_file('wmh_positive_tweets_wc.png')


# # WordCloud - Negative Tweets

# In[45]:


# WordCloud - Negative Tweets

negative_wordcloud = WordCloud(mask=mask,width=900, height=500).generate(wordcloud(negative_tweets)[0])

plot_Cloud(negative_wordcloud)
negative_wordcloud.to_file('wmh_negative_tweets_wc.png')


# # WordCloud - Neutral Tweets

# In[46]:


# WordCloud - Neutral Tweets

neutral_wordcloud = WordCloud(background_color= "white",mask=mask,width=900, height=500).generate(wordcloud(neutral_tweets)[0])

plot_Cloud(neutral_wordcloud)
neutral_wordcloud.to_file('wmh_neutral_tweets_wc.png')


# # Wordcloud - All Tweets

# In[47]:


# Wordcloud - All Tweets

total_wordcloud = WordCloud(mask=mask, width=900, height=500).generate(wordcloud(dfs)[0])

plot_Cloud(total_wordcloud)
total_wordcloud.to_file('wmh_total_tweets_wc.png')


# # Most Frequent Words - Total Tweets

# In[48]:


# Most Frequent Words - Total Tweets

aa = nltk.FreqDist(wordcloud(dfs)[1])
dd = pd.DataFrame({'Wordcount': list(aa.keys()),
                  'Count': list(aa.values())})
# selecting top 10 most frequent hashtags     
dd = dd.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,10))
plt.title('Most Frequent Words in all of the Tweets')
ax = sns.barplot(data=dd,palette="Purples_r", x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
fig = ax.get_figure()
fig.savefig("wmh_total_tweets_wf.png")
plt.show()


# # Most Frequent Words - Positive Tweets

# In[49]:


# Most Frequent Words - Positive Tweets

ap = nltk.FreqDist(wordcloud(positive_tweets)[1])
dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
# selecting top 10 most frequent hashtags     
dp = dp.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,10))
plt.title('Most Frequent Words in the Positive tweets')
ax = sns.barplot(data=dp,palette="rocket", x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
fig = ax.get_figure()
fig.savefig("wmh_positive_tweets_wf.png")
plt.show()


# # Most Frequent Words - Negative Tweets

# In[50]:


# Most Frequent Words - Negative Tweets

an = nltk.FreqDist(wordcloud(negative_tweets)[1])
dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
# selecting top 10 most frequent hashtags     
dn = dn.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,10))
plt.title('Most Frequent Words in the Negative tweets')
ax = sns.barplot(data=dn,palette="gray", x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
fig = ax.get_figure()
fig.savefig("wmh_negative_tweets_wf.png")
plt.show()


# # Most Frequent Words - Neutral Tweets

# In[51]:


# Most Frequent Words - Neutral Tweets

au = nltk.FreqDist(wordcloud(neutral_tweets)[1])
du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
# selecting top 10 most frequent hashtags     
du = du.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,10))
plt.title('Most Frequent Words in the Neutral tweets')
ax = sns.barplot(data=du, palette= "Blues_d",x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
fig = ax.get_figure()
fig.savefig("wmh_neutral_tweets_wf.png")
plt.show()


# In[ ]:




