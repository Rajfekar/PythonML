import numpy as np
import pandas as pd
column =  ['Eleanor','Chidi','Tahani','Json']
data = np.random.randint(low=0,high=100,size=(3,4)); 
df = pd.DataFrame(data, columns=column);
print(df);
print(df['Eleanor'][1]);
df["jset"] = df['Tahani']+df['Json']
# df2 = df  #REFRENCE COPY
df2 = df.copy()
print(df,"\n",df2)
print("starting value of df: ", df["Json"][1])
print("starting value of df2: ", df2["Json"][1])

df2.at[1,"Json"] = df2["Json"][1] +5

print("starting value of df: ", df2["Json"][1])
print("starting value of df2: ", df["Json"][1])