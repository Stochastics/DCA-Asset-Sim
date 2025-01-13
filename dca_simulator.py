import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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

def plot_portfolio_value(detailed_metrics, scenario_name):
    """
    Plots the portfolio value over time for a given scenario.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=detailed_metrics, x='Date', y='Portfolio Value', label=scenario_name)
    plt.title(f"Portfolio Value Over Time - {scenario_name}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_weekly_returns(detailed_metrics, scenario_name):
    """
    Plots the weekly returns for a given scenario.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=detailed_metrics, x='Date', y='Weekly Return (%)', label=scenario_name)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Weekly Returns - {scenario_name}")
    plt.xlabel("Date")
    plt.ylabel("Weekly Return (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_cumulative_return(detailed_metrics, scenario_name):
    """
    Plots the cumulative return over time for a given scenario.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=detailed_metrics, x='Date', y='Cumulative Return (%)', label=scenario_name)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Cumulative Return Over Time - {scenario_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_combined_metrics(all_summaries, metrics, title, output_dir=None):
    """
    Plots a grouped bar chart comparing key metrics across scenarios.

    Parameters:
    - all_summaries (list of dicts): Summary metrics for each scenario.
    - metrics (list of str): List of metric names to compare (e.g., 'Total Return (%)').
    - title (str): Title of the plot.
    - output_dir (str, optional): Directory to save the plot. If None, the plot is displayed.
    """
    # Convert summaries to a DataFrame
    df = pd.DataFrame(all_summaries)
    df.set_index("Scenario", inplace=True)
    df = df[metrics]
    ax = df.plot(kind="bar", figsize=(10, 6), edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Scenario")
    ax.legend(title="Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_dir:
        filename = f"{output_dir}/combined_metrics_comparison.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    else:
        plt.show()


def plot_combined_metrics_panel(all_summaries, scenarios, metrics, output_dir=None):
    """
    Creates a 2x2 panel comparison for selected metrics across all scenarios.

    Parameters:
    - all_summaries (list of dicts): Summary metrics for each scenario.
    - scenarios (list of dicts): Scenario definitions for labels.
    - metrics (list of str): Metrics to plot in the panel.
    - output_dir (str, optional): Directory to save the plot. If None, the plot is displayed.
    """

    df = pd.DataFrame(all_summaries)
    df["Scenario"] = [s["name"] for s in scenarios]
    df.set_index("Scenario", inplace=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        df[metric].sort_values().plot(kind="bar", ax=axes[i], color="skyblue", edgecolor="black")
        axes[i].set_title(metric)
        axes[i].set_ylabel("Value")
        axes[i].tick_params(axis="x", labelrotation=0, labelsize=8)

    plt.tight_layout()

    if output_dir:
        filename = f"{output_dir}/metrics_panel.png"
        plt.savefig(filename)
        print(f"Metrics panel saved to {filename}")
    else:
        plt.show()

def dca_asset_simulator(config_path, total_weekly_investment=None, years=None, output_dir=None):
    """
    Simulates DCA across scenarios from a given config file and generates plots.

    Parameters:
    - config_path (str): Path to the YAML configuration file.
    - total_weekly_investment (float, optional): Override the default weekly investment amount.
    - years (int, optional): Override the default number of years.
    - output_dir (str, optional): Directory to save the comparison plots. If None, plots are displayed.

    Returns:
    - all_summaries (list): Summary metrics for each scenario.
    - all_detailed_metrics (list): Detailed metrics for each scenario.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        with open(config_path, "r") as config_file:
            scenarios = yaml.safe_load(config_file).get("scenarios", [])
            if not scenarios:
                raise ValueError("No scenarios found in the configuration file.")
    except Exception as e:
        logging.error(f"Error reading configuration file: {e}")
        return [], []

    all_detailed_metrics = []
    all_summaries = []

    for scenario in scenarios:
        logging.info(f"Running scenario: {scenario['name']}")
        print(format_assets_and_weights(scenario['assets'], scenario['weights']))

        try:
            summary, detailed_metrics = process_scenario(scenario, total_weekly_investment, years)
        except Exception as e:
            logging.error(f"Error processing scenario '{scenario['name']}': {e}")
            continue

        all_summaries.append(summary)
        all_detailed_metrics.append(detailed_metrics)

        logging.info("\nSummary Results:")
        for key, value in summary.items():
            logging.info(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")

        logging.info("-" * 40)

    metrics_to_compare = [
        "Total Return (%)",
        "Annualized Return (%)",
        "Maximum Drawdown (%)",
        "Average Volatility (%)"
    ]

    plot_combined_metrics_panel(all_summaries, scenarios, metrics_to_compare, output_dir)
    save_combined_metrics(all_detailed_metrics)

    return all_summaries, all_detailed_metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCA Asset Simulator")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--total_weekly_investment", type=float, default=None, help="Weekly investment override.")
    parser.add_argument("--years", type=int, default=None, help="Override for number of years.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save comparison plots.")
    args = parser.parse_args()

    dca_asset_simulator(
        config_path=args.config,
        total_weekly_investment=args.total_weekly_investment,
        years=args.years,
        output_dir=args.output_dir
    )
    import argparse

    parser = argparse.ArgumentParser(description="DCA Asset Simulator")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--total_weekly_investment", type=float, default=None, help="Weekly investment override.")
    parser.add_argument("--years", type=int, default=None, help="Override for number of years.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save comparison plots.")
    args = parser.parse_args()

    dca_asset_simulator(
        config_path=args.config,
        total_weekly_investment=args.total_weekly_investment,
        years=args.years,
        output_dir=args.output_dir
    )