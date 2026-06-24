MAANG_TICKERS = {
    'META': 'Meta',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'NFLX': 'Netflix',
    'GOOGL': 'Google'
}

def get_ticker_name(
    ticker: str
    ) -> str:
    return MAANG_TICKERS.get(ticker, ticker)