
import plotly.express as px
import pandas as pd


stages = ["访问数", "下载数", "注册数", "搜索数", "付款数"] # 漏斗的阶段
# 漏斗的数据

df_male = pd.DataFrame(dict(number=[30, 15, 10, 6, 1], stage=stages))
df_male['性别'] = '男'
df_female = pd.DataFrame(dict(number=[29,17,8,3,1],stage=stages))
df_female['性别'] = "女"

df = pd.concat([df_male, df_female], axis=0) # 把男生女生的数据连接至一个新的Dataframe对象df
fig = px.funnel(df, x='number', y='stage', color='性别') # 把df中的数据传进漏斗
fig.show()

print(df)
