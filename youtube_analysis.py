#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt  


# In[4]:


comments = pd.read_csv(r'E:\Placement 2022-23\Analytics\projects\Youtube_project_shan_singh\Youtube_project_shan_singh\UScomments.csv',error_bad_lines=False)


# In[5]:


# checking the first or top values in the file 
comments.head()


# In[6]:


#checking if there are any empty values 
comments.isnull().sum()


# In[7]:


#as we have 25 null values,removing it for no further confusion during analyses 
comments.dropna(inplace=True)


# In[8]:


#checking empty values again 
comments.isnull().sum()


# In[12]:


get_ipython().system('pip install textblob')


# In[9]:


from textblob import TextBlob 


# In[10]:


#performing sentiment analysis on one comment. 
TextBlob('MY FAN . attendance').sentiment 


# In[11]:


#now to perform the sentiment analysis, more like checking the polarity on just a sample space--
#slicing 
df=comments[0:10000]


# In[12]:


polarity = []
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[13]:


#chceking polarity for the first 10 values 
polarity[0:10]


# In[14]:


#adding new feature/column into the dataset. 
comments['polarity'] = polarity 
comments.head()


# In[15]:


#filtering the positive and negative comments 
comments_positive = comments[comments['polarity']==1]


# In[16]:


comments_positive 


# In[17]:


comments_negative = comments[comments['polarity']==-1]


# In[18]:


comments_negative 


# In[19]:


#performing the wordcloud on sentiment analysis. 


# In[20]:


from wordcloud import WordCloud, STOPWORDS 


# In[21]:


#in order to proceed further, we have to convert data to string nature
comments_negative['comment_text']


# In[22]:


total_negative_comments = ' '.join(comments_negative['comment_text'])


# In[23]:


total_negative_comments 


# In[24]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_negative_comments)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off') 


# In[26]:


comments_positive['comment_text']


# In[27]:


total_positive_comments = ' '.join(comments_positive['comment_text']) 


# In[28]:


total_positive_comments  


# In[29]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_positive_comments)
plt.figure(figsize=(7,5))
plt.imshow(wordcloud)
plt.axis('off') 


# In[30]:


positive_comments_list = []
for comment in comments_positive['comment_text']:
    positive_comments_list.append(comment)


# In[31]:


len(positive_comments_list)


# In[32]:


positive_comments_list[0:10]


# In[33]:


from collections import Counter 


# In[34]:


Counter(positive_comments_list).most_common(5) 


# In[35]:


pcomments = [Counter(positive_comments_list).most_common(5)[i][0] for i in range(5)]


# In[36]:


pfreqs = [Counter(positive_comments_list).most_common(5)[i][1] for i in range(5)]


# In[37]:


pcomments


# In[38]:


pfreqs


# In[39]:


get_ipython().system('pip install plotly ')


# In[40]:


import plotly.graph_objs as go 


# In[41]:


from plotly.offline import iplot


# In[42]:


ptrace = go.Bar(x=pcomments,y=pfreqs)


# In[127]:


iplot([ptrace])


# In[44]:


#collecting entire data of youtube 
import os 


# In[45]:


path =r'E:\Placement 2022-23\Analytics\projects\Youtube_project_shan_singh\Youtube_project_shan_singh\additional_data'


# In[46]:


files = os.listdir(path)


# In[47]:


files 


# In[48]:


#extracting only csv files 
for i in range(1,len(files),2):
    print(i)


# In[49]:


files_csv=[files[i] for i in range(0,len(files),2)]


# In[50]:


files_csv


# In[51]:


files_csv[0].split('.')[0][0:2]


# In[52]:


full_df=pd.DataFrame()

for file in files_csv:
    current_df=pd.read_csv(path+'/'+file,encoding='iso-8859-1',error_bad_lines=False)
    
    current_df['country']=file.split('.')[0][0:2]
    full_df=pd.concat([full_df,current_df])


# In[53]:


full_df.head()


# In[54]:


full_df.shape


# In[87]:


#analysing the category that got the most likes 


# In[86]:


#as there is no category column, we extract the data from the given and then map it in order to analyse with the likes column 
full_df['category_id'].unique()


# In[60]:


catf=pd.read_csv('E:\Placement 2022-23\Analytics\projects\Youtube_project_shan_singh\Youtube_project_shan_singh\category_file.txt',sep=':')


# In[62]:


catf.reset_index(inplace=True)


# In[63]:


catf


# In[65]:


catf.columns=['Category_id','Category_name']


# In[66]:


catf.set_index('Category_id',inplace=True)


# In[67]:


catf


# In[69]:


dct=catf.to_dict()


# In[71]:


dct['Category_name']


# In[73]:


#mapping this dictionary on the category id feature 


# In[74]:


full_df['Category_name'] = full_df['category_id'].map(dct['Category_name'])


# In[75]:


full_df.columns


# In[76]:


full_df.head()


# In[77]:


#distribution of likes on each category 


# In[79]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Category_name',y='likes',data=full_df)
plt.xticks(rotation='vertical')


# In[88]:


#to analyse whether the audience is engaged or not 
#checking the like rate, comment rate, dislike rate 


# In[91]:


full_df.columns
#in this views, likes, dislikes, comment_count play important role 


# In[92]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] =(full_df['dislikes']/full_df['views'])*100
full_df['coomment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[93]:


full_df.head(2)


# In[94]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Category_name',y='like_rate',data=full_df)
plt.xticks(rotation='vertical')


# In[95]:


sns.regplot(data=full_df,x='views',y='likes')


# In[96]:


cor = full_df[['views','likes','dislikes']].corr()


# In[97]:


cor


# In[98]:


sns.heatmap(cor,annot=True)


# In[99]:


#analysing the trending videos that is which channel have the most trending videos or the most number of videos 


# In[100]:


full_df.head(2)


# In[120]:


cdf = full_df.groupby('channel_title')['video_id'].count().sort_values(ascending=False).to_frame().reset_index()


# In[123]:


cdf.rename(columns={'video_id':'total_videos'})


# In[117]:


#let us try something new now-- bar chart using plt 


# In[124]:


import plotly.express as px 


# In[126]:


px.bar(data_frame=cdf[0:20],x='channel_title',y='video_id')


# In[140]:


bargraph = cdf.plot.bar(x='channel_title',y='video_id',fontsize='9')


# In[131]:


#checking if punctuation plays a role or effects in the dislikes likes and comments of the video 


# In[132]:


import string 


# In[133]:


string.punctuation 


# In[134]:


def punc_count(x):
    return len([c for c in x if c in string.punctuation])


# In[135]:


full_df['title'][0] 


# In[136]:


text='Eminem - Walk On Water (Audio) ft. BeyoncÃ©'


# In[137]:


punc_count(text)


# In[143]:


sample=full_df[0:10000]


# In[144]:


sample['count_punc']=sample['title'].apply(punc_count)


# In[145]:


sample.head()


# In[146]:


#creating box plot with count punc and views 


# In[147]:


plt.figure(figsize=(12,8))
sns.boxplot(x='count_punc',y='views',data=sample)


# In[152]:


cor_of_sample = sample['count_punc'].corr(sample['dislikes'])


# In[153]:


cor_of_sample 


# In[ ]:




