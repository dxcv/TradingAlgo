import json
import numpy as np
import os
import pandas as pd
import urllib
import math

# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1546300800&end=9999999999&period=300&resolution=auto'

# parse json returned from the API to Pandas DF
openUrl = urllib.request.urlopen(url)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())
df = pd.DataFrame(d)

original_columns=[u'date', u'close',  u'high', u'low', u'open', u'volume']
new_columns = ['Timestamp','Close','High','Low','Open','Volume']
df = df.loc[:,original_columns]
df.columns = new_columns
df.to_csv('bitcoin201901to201905.csv',index=None)

df = df.set_index('Timestamp')
df.head()
