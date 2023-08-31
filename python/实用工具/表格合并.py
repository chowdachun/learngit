

import pandas as pd


df_ls =[]
for i in range(3):
    a = i+1
    path = f"/Users/admin/Desktop/AAAA/{a}.csv"
    print(path)
    df = pd.read_csv(path)
    print(df.shape)
    df_ls.append(df)
print("df_ls:",len(df_ls))

df = pd.concat(df_ls)
print("df_concat:",df.shape)


df.to_csv("/Users/admin/Desktop/AAAA/data.csv",index=False)