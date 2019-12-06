#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import csv
from scipy.spatial.distance import cityblock
from sklearn.metrics import roc_curve


# In[3]:


data = pd.read_csv("Collecting_keyStorke.csv")
userList = data.user.unique()
keyList = data.key.unique()
df = pd.DataFrame(columns=['subject','key','H','UD','DD'])
for i in range(0, len(userList)):
    for j in range(0,len(keyList)):
        queryData = data.query("user=='" +userList[i]+ "' and key==" + str(keyList[j]) + " and key >=33 and key<=122")
        queryLen = len(queryData)
        finalData = {}
        if queryLen > 0:
            if(queryLen > 2):
                for k in range(0,queryLen,2):
                    finalData['subject'] = userList[i]
                    finalData['key'] = chr(keyList[j])
                    finalData['H'] = (int(queryData.iloc[k+1].Time) - int(queryData.iloc[k].Time))/1000
                    keyUpIndex = queryData.iloc[k+1].name
                    if(data.iloc[keyUpIndex + 1].user == userList[i]):
                        finalData['UD'] = (int(data.iloc[keyUpIndex+1].Time) - int(queryData.iloc[k+1].Time))/1000
                        finalData['DD'] = (int(data.iloc[keyUpIndex+1].Time) - int(queryData.iloc[k].Time))/1000
                    else:
                        finalData['UD'] =  finalData['H']
                        finalData['DD'] = finalData['H']
                    df = df.append(finalData,ignore_index=True )
            else:
                finalData['subject'] = userList[i]
                finalData['key'] = chr(keyList[j])
                finalData['H']= (int(queryData.query("keyEvent=='Up'").Time) - int( queryData.query("keyEvent=='Down'").Time))/1000
                keyUpIndex = queryData.query("keyEvent=='Up'").index[0]
                if(data.iloc[keyUpIndex + 1].user == userList[i]):
                        finalData['UD'] = (int(data.iloc[keyUpIndex+1].Time) - int( queryData.query("keyEvent=='Up'").Time))/1000
                        finalData['DD'] = (int(data.iloc[keyUpIndex+1].Time) - int( queryData.query("keyEvent=='Down'").Time))/1000
                else:
                    finalData['UD'] =  finalData['H']
                    finalData['DD'] =  finalData['H']
                df = df.append(finalData,ignore_index=True )
            
           
                
f = open("KeyStrokeDistance.csv", 'w',newline='\n')
writer = csv.writer(f)
writer.writerow(['subject','key','H','UD','DD'])
for row in df.iterrows():
    #print(row[1])
    writer.writerow(row[1])
    
f.close()
print(df[df.subject == 'rakshith'])


# In[9]:


groupedDf = df.query("subject=='rakshith'").groupby(['subject','key','H','UD','DD']).all()
groupedDf


# In[ ]:





# In[ ]:




