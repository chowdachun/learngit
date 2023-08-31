
import pandas as pd
import datetime as dt



end = pd.to_datetime('2023-07-24')
stat = pd.to_datetime('2023-08-9')

diff = (stat - end).days / 7

print(diff)