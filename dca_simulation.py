import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse

def fetch_data(assets, start_date, end_date):
    """
    Fetches historical data for the given assets.
    """
    data = yf.download(assets, start=start_date, end=end_date, interval='1wk')
    try:
        data = data['Close']
    except KeyError:
        raise KeyError("No 'Close' column found in the data. Check the data structure or asset tickers.")
    data.ffill(inplace=True)
    return data

def calculate_portfolio_value(data, assets, allocations, total_weekly_investment):
    """
    Calculates portfolio values and metrics for each date.
    """
    portfolio_values = []
    cumulative_units = {asset: 0.0 for asset in assets}
    total_investment = 0.0
    metrics = []

    for date in data.index:
        weekly_portfolio_value = 0.0

        for asset, allocation in zip(assets, allocations):
            if allocation == 0:
                continue
            price = data.loc[date, asset]
            if not pd.isna(price):
                weekly_investment = total_weekly_investment * allocation
                units_purchased = weekly_investment / price
                cumulative_units[asset] += units_purchased
                total_investment += weekly_investment

            weekly_portfolio_value += cumulative_units[asset] * price if not pd.isna(price) else 0.0

        portfolio_values.append(weekly_portfolio_value)
        weekly_return = (
            (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2] * 100
            if len(portfolio_values) > 1 else 0.0
        )
        cumulative_return = ((portfolio_values[-1] - total_investment) / total_investment * 100)

        metrics.append({
            'Date': date,
            'Portfolio Value': weekly_portfolio_value,
            'Weekly Return (%)': weekly_return,
            'Cumulative Return (%)': cumulative_return,
            'Total Investment': total_investment
        })

    return portfolio_values, metrics

def calculate_summary_metrics(portfolio_values, total_investment, years):
    """
    Calculates summary metrics such as total return, max drawdown, and volatility.
    """
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.cummax()
    drawdowns = (portfolio_series - running_max) / running_max
    max_drawdown = drawdowns.min() * 100

    final_value = portfolio_values[-1] if portfolio_values else 0
    total_return = (final_value - total_investment) / total_investment
    annualized_return = (1 + total_return) ** (1 / years) - 1

    weekly_returns = portfolio_series.pct_change().dropna()
    annualized_volatility = weekly_returns.std() * (52 ** 0.5) * 100

    return {
        'Total Investment': total_investment,
        'Final Value': final_value,
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Maximum Drawdown (%)': max_drawdown,
        'Average Volatility (%)': annualized_volatility,
        'Number of Years': years,
    }

def process_scenario(scenario, total_weekly_investment=None, years=None):
    """
    Processes a single scenario and returns its summary and detailed metrics.
    """
    assets = scenario['assets']
    weights = scenario['weights']
    years = years if years is not None else scenario.get('years', 5)
    total_weekly_investment = total_weekly_investment if total_weekly_investment is not None else scenario.get('total_weekly_investment', 100)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = fetch_data(assets, start_date, end_date)
    portfolio_values, metrics = calculate_portfolio_value(data, assets, weights, total_weekly_investment)
    summary = calculate_summary_metrics(portfolio_values, metrics[-1]['Total Investment'], years)
    detailed_metrics = pd.DataFrame(metrics)
    detailed_metrics['Scenario'] = scenario['name']

    return summary, detailed_metrics

def save_combined_metrics(all_detailed_metrics, filename="combined_detailed_metrics.csv"):
    """
    Combines all detailed metrics and saves to a CSV file.
    """
    combined_metrics = pd.concat(all_detailed_metrics, ignore_index=True)
    combined_metrics.to_csv(filename, index=False)
    print(f"\nAll scenarios' detailed metrics have been saved to '{filename}'.")

def format_assets_and_weights(assets, weights):
    """
    Formats assets and weights into a pretty string.
    """
    output = ["Portfolio Composition:"]
    for asset, weight in zip(assets, weights):
        output.append(f"  - {asset}: {weight * 100:.2f}%")
    return "\n".join(output)

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate DCA for various scenarios.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--total_weekly_investment", type=float, default=None, help="Total weekly investment amount to override the scenario defaults.")
    parser.add_argument("--years", type=int, default=None, help="Number of years to override the scenario defaults.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.config, "r") as config_file:
        scenarios = yaml.safe_load(config_file)["scenarios"]

    all_detailed_metrics = []

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        print(format_assets_and_weights(scenario['assets'], scenario['weights']))

        summary, detailed_metrics = process_scenario(scenario, args.total_weekly_investment, args.years)

        all_detailed_metrics.append(detailed_metrics)

        print("\nSummary Results:")
        for key, value in summary.items():
            print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")

        print("-" * 40)

    save_combined_metrics(all_detailed_metrics)