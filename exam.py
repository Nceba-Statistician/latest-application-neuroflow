import pandas as pd
import mplfinance as mpf

# Assuming you have your data loaded into a DataFrame like this:
# df = pd.DataFrame({
#     'Date': ['2020.01.01', '2020.01.01', ...],
#     'Time': ['00:00:00', '00:30:00', ...],
#     'Open': [108.740, 108.697, ...],
#     'High': [108.748, 108.701, ...],
#     'Low': [108.604, 108.643, ...],
#     'Close': [108.681, 108.669, ...]
# })
df = ""
# 1. Combine Date and Time and set as index
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('DateTime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close']] # Keep only necessary columns

# 2. Plot candlesticks
mpf.plot(df, type='candle', style='yahoo',
         title='Candlestick Chart',
         ylabel='Price')