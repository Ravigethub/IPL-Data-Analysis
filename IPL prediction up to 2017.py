#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[44]:


match=pd.read_csv("matches.csv")
delivery=pd.read_csv("deliveries.csv")


# In[45]:


match.head()


# In[46]:


match.shape


# In[47]:


delivery.head()


# In[48]:


delivery.shape


# In[50]:


total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[51]:


total_score_df=total_score_df[total_score_df.inning==1]


# In[52]:


total_score_df


# In[54]:


match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[56]:


match_df


# In[58]:


match_df['team1'].unique()


# In[60]:


match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[62]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[63]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=  match_df[match_df['team2'].isin(teams)]


# In[65]:


match_df.shape


# In[68]:


match_df=match_df[match_df['dl_applied']==0]


# In[69]:


match_df=match_df[['match_id','city','winner','total_runs']]


# In[70]:


delivery_df=match_df.merge(delivery,on='match_id')


# In[76]:


delivery_df


# In[79]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[85]:


delivery_df


# In[86]:


delivery_df['current_score']= delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[91]:


delivery_df['runs_left']= delivery_df['total_runs_x']-delivery_df['current_score']


# In[93]:


delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])


# In[96]:


delivery_df


# In[95]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna(0)


# In[100]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == 0 else 1)


# In[102]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype(int)


# In[103]:


wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values


# In[106]:


delivery_df['wickets']=10-wickets


# In[113]:


delivery_df['curr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[114]:


delivery_df['rrr']=(delivery_df['current_score']*6)/delivery_df['balls_left']


# In[119]:


delivery_df


# In[117]:


def result(row):
    return 1 if row['batting_team']==row['winner'] else 0


# In[118]:


delivery_df['result']=delivery_df.apply(result, axis=1)


# In[133]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','curr','rrr','result']]


# In[137]:


final_df = final_df.sample(final_df.shape[0])


# In[ ]:





# In[140]:


final_df


# In[141]:


final_df.dropna(inplace=True)
    


# In[143]:


final_df = final_df[final_df['balls_left'] != 0]
final_df


# In[144]:


x=final_df.iloc[:,:-1]


# In[145]:


y=final_df.iloc[:,-1]


# In[147]:


from sklearn.model_selection import train_test_split


# In[148]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[149]:


x_train


# In[151]:


y_train


# In[153]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[156]:


trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False, drop='first'),
                         ['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[157]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[181]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[182]:


pipe.fit(x_train,y_train)


# In[183]:


y_pred=pipe.predict(x_test)
y_pred


# In[184]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[185]:


pipe.predict_proba(x_test)[10]


# In[186]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[187]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','curr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[188]:


temp_df,target = match_progression(delivery_df,1,pipe)
temp_df


# In[189]:


plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[190]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




