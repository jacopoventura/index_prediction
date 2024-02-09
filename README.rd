This project aims at building a ML model capable to predict the price movement of a stock market ticker.
I want a ML-based system that:
1- predict if the next day is negative
2- predict the probabilities of change for each class (>0.5, 0, -0.5, -1, -2)
3- predict the price

Given N the number of days of the future horizon.



Sources to read:
1. Financial TA in python: https://thepythoncode.com/article/introduction-to-finance-and-technical-indicators-with-python
2. Tutorial RNN: https://thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras?utm_content=cmp-true
3. https://github.com/DestrosCMC/Forecasting-SPY/blob/main/README.md

NOTE: the starting scoring is 0.5 (coin toss). everything better than 0.53 (% positive days) is an improvement.

TO DO:
1. use StockDataFrame (https://github.com/jealous/stockstats/blob/master/README.md)
    - calculate all the indicators first
    - select specific columns and train different Random Forest models (with relative change and with full price)
2. clean and add more scores RandomForest
3. Understand and clean data preparation RNN
4. add EURO/USD exchange rate and train
5. Python tutorial RNN
6. use normalization in Random Forest ?