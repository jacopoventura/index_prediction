This project builds a ML model capable to predict the price movement of a stock market ticker.
The business logic of the product (query data, preparation of data ans model usage) is also provided in the main script.

The ML model shall:
1. predict the price the next N days (number of days of the future horizon)
2. N is a configurable parameter

We are option traders, thus we are particularly interested in predicting negative days for two reasons:
1. sell put to be open in negative days
2. 0DTE strategies
3. avoid rolling of naked puts

Sources to read:
1. Financial TA in python: https://thepythoncode.com/article/introduction-to-finance-and-technical-indicators-with-python
2. Tutorial RNN: https://thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras?utm_content=cmp-true
3. https://github.com/DestrosCMC/Forecasting-SPY/blob/main/README.md

NOTE: the starting scoring is 0.5 (coin toss). everything with precision better than 0.53 (% positive days) is an improvement.
