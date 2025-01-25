import yfinance as yf

# Retrieve option data for a stock
ticker = yf.Ticker('AAPL')

# Get option chain for nearest expiration
options = ticker.option_chain()

# Get historical stock data
stock_history = ticker.history(period="1y")

# Example of accessing call and put options
calls = options.calls
puts = options.puts

print(calls)
print(puts)
