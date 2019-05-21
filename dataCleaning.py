import pandas as pd
import numpy as np
import nltk as nl
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#DataFrame
df_main=pd.read_csv(r'D:\DataAnalysis\Sparsh\FinalProj\youtube-new\USvideos.csv', header=0)
#Copy dataframe
df_copy=df_main.copy()


#creating new Date field:
x=np.zeros((len(df_main),1))
z=np.zeros((len(df_main),1))
dd=np.zeros((len(df_main),1))
md=np.zeros((len(df_main),1))
yd=np.zeros((len(df_main),1))
for i in range(len(df_main)):
    tm=df_main['trending_date'][i].split('.')[2]
    ty=df_main['trending_date'][i].split('.')[0]
    td=df_main['trending_date'][i].split('.')[1]
    pd=df_copy['publish_time'][i].split('T')
    pubd=pd[0]
    l=pubd.split('-')
    py=l[0][2:4]
    pm=l[1]
    pd=l[2]
    pubt=pd[1][0:2]
    pt=pubt[0:2]
    #print(ty,'-',py,'\n',tm,'-',pm,'\n',td,'-',pd,'\n\n\n')
    dd[i]=int(td)-int(pd)
    md[i]=int(tm)-int(pm)
    yd[i]=int(ty)-int(py)
    #print('The difference in the year between trending and publishing is:',int(ty)-int(py))
    #print('The difference in the month between trending and publishing is:',int(tm)-int(pm))
    #print('The difference in the day between trending and publishing is:',int(td)-int(pd))



sid = SentimentIntensityAnalyzer()
nl.download('vader_lexicon')

#Sentiment Analysis of Title
negTitle=np.zeros((len(df_main),1))
neuTitle=np.zeros((len(df_main),1))
posTitle=np.zeros((len(df_main),1))
compdTitle=np.zeros((len(df_main),1))
for i in range(len(df_main)):
    psTitle=sid.polarity_scores(df_main['title'][i])
    negTitle[i]=psTitle['neg']
    neuTitle[i]=psTitle['neu']
    posTitle[i]=psTitle['pos']
    compdTitle[i]=psTitle['compound']

negTg=np.zeros((len(df_main),1))
neuTg=np.zeros((len(df_main),1))
posTg=np.zeros((len(df_main),1))
compdTg=np.zeros((len(df_main),1))
for i in range(len(df_main)):
    psTags=sid.polarity_scores(df_main['tags'][i])
    negTg[i]=psTags['neg']
    neuTg[i]=psTags['neu']
    posTg[i]=psTags['pos']
    compdTg[i]=psTags['compound']

df_copy['negTitle']=negTitle
df_copy['neuTitle']=neuTitle
df_copy['posTitle']=posTitle
df_copy['compdTitle']=compdTitle
df_copy['negTg']=negTg
df_copy['neuTg']=neuTg
df_copy['posTg']=posTg
df_copy['compdTg']=compdTg


    

def change_to_num(oldColName,newColName,df):
    x=df[oldColName].unique()
    CTN={name:num for num,name in enumerate(x)}
    df[newColName]=df[oldColName].replace(CTN)
    return (df[newColName])

cont=change_to_num('channel_title','title_num',df_copy)

df_copy['PD']=pd
df_copy['PM']=pm
df_copy['PY']=py
df_copy['PT']=pubt
df_copy['TD']=td
df_copy['TM']=tm
df_copy['TY']=ty

cols=['category_id', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled', 'ratings_disabled','video_error_or_removed', 'negTitle', 'neuTitle','posTitle', 'compdTitle', 'negTg', 'neuTg', 'posTg', 'compdTg','title_num', 'PD', 'PM', 'PY', 'PT', 'TD', 'TM', 'TY']

indexNum=df_copy[df_copy['ratings_disabled']==True].index
df_ratingEnabled=df_copy[cols]
df_ratingEnabled.drop(indexNum,inplace=True)
df_ratingEnabled.to_csv(r'D:\DataAnalysis\Sparsh\FinalProj\ratingEnabled.csv')

indexNum1=df_copy[df_copy['comments_disabled']==True].index
df_commentsEnabled=df_copy[cols]
df_commentsEnabled.drop(indexNum1,inplace=True)
df_commentsEnabled.to_csv(r'D:\DataAnalysis\Sparsh\FinalProj\commentEnabled.csv')    
