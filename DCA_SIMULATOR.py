import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_dca(assets, allocations, total_weekly_investment, years):
    """
    Simulates DCA for the given assets over the past N years with percentage-based allocations.

    Parameters:
    - assets (list): List of asset tickers.
    - allocations (list): Percentage allocation for each asset (must sum to 1.0 or less).
    - total_weekly_investment (float): Total amount to invest weekly across all assets.
    - years (int): Number of past years to simulate.

    Returns:
    - DataFrame: Annualized returns for each year and overall returns.
    """
    if len(assets) != len(allocations):
        raise ValueError("Allocations must be the same length as assets.")

    if sum(allocations) > 1.0:
        raise ValueError("Allocations must sum to 1.0 or less.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = yf.download(assets, start=start_date, end=end_date, interval='1wk')
    try:
        data = data['Close']
    except KeyError:
        raise KeyError("No 'Close' column found in the data. Check the data structure or asset tickers.")

    data.ffill(inplace=True)

    investment_summary = {}
    for asset, allocation in zip(assets, allocations):
        if allocation == 0:  # Skip assets with 0 allocation
            continue

        weekly_investment = total_weekly_investment * allocation
        cumulative_units = 0.0
        total_investment = 0.0
        values_over_time = []

        for date, price in data[asset].items():
            if not pd.isna(price):
                units_purchased = weekly_investment / price
                cumulative_units += units_purchased
                total_investment += weekly_investment
                values_over_time.append(cumulative_units * price)

        final_value = values_over_time[-1] if values_over_time else 0
        total_return = (final_value - total_investment) / total_investment
        annualized_return = (1 + total_return) ** (1 / years) - 1

        investment_summary[asset] = {
            'Total Investment': total_investment,
            'Final Value': final_value,
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
        }

    summary_df = pd.DataFrame.from_dict(investment_summary, orient='index')
    return summary_df


# Define assets, allocations, and other parameters
assets = ['BTC-USD', 'QQQ', 'VTI']
allocations = [0.33, 0.33, 0.33]  # 50% BTC, 50% QQQ, 0% VTI
total_weekly_investment = 1000  # Total $100 invested weekly
years = 8  # Simulate for the past 5 years

# Run the simulation
results = simulate_dca(assets, allocations, total_weekly_investment, years)
print("DCA Simulation Results:")
print(results)