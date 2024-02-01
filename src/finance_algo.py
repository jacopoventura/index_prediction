import pandas as pd


# Calculate RSI
def calculate_rsi(prices: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the RSI given the historical price dataset.
    :param prices: Dataframe's single column with the history of the price
    :param period: period of the Relative Strength Index
    :return: Relative Strength Index
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
