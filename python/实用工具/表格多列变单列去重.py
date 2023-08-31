
import pandas as pd
import numpy as np
import math

path = "/Users/admin/Desktop/兴宁4.6-5.3用户复乘情况.xlsx"
df = pd.read_excel(path,sheet_name=4)

# print(df.head(20))
# print(df.shape)
# print(df.columns)

list_l =[]
for a in np.arange(df.shape[1]):
    df_1 = df.iloc[3:,a]
    ls = df_1.tolist()
    list_l = list_l  + ls

new_list_1 = list(set(list_l ))
new_list =  [x for x in new_list_1 if not math.isnan(x)]

print(len(new_list))
print(new_list)

s = pd.Series(new_list,name="id")
s.to_csv("/Users/admin/Desktop/s.csv",index=False)