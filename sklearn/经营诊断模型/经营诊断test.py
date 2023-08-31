
import pandas as pd

after_end = '2023-07-31'
after_star = '2023-07-24'
input_end = pd.to_datetime(after_end, format='%Y-%m-%d')
input_star = pd.to_datetime(after_star, format='%Y-%m-%d')

day = (input_end-input_star).days
print(day)