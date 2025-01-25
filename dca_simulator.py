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

def save_combined_metrics(all_detailed_metrics, 
                          folder,
                          filename="combined_detailed_metrics.csv"):
    """
    Combines all detailed metrics and saves to a CSV file.
    """
    combined_metrics = pd.concat(all_detailed_metrics, ignore_index=True)
    combined_metrics.to_csv(folder + "/" + filename, index=False)
    print(f"\nAll scenarios' detailed metrics have been saved to '{filename}'.")

def format_assets_and_weights(assets, weights):
    """
    Formats assets and weights into a pretty string.
    """
    output = ["Portfolio Composition:"]
    for asset, weight in zip(assets, weights):
        output.append(f"  - {asset}: {weight * 100:.2f}%")
    return "\n".join(output)

def analyze_asset_correlation(assets, start_date, end_date, window_size='30D', data_freq='1D'):
    """
    Analyzes the correlation between two assets over time using a rolling window.
    
    Parameters:
    - assets (list): List of two asset tickers to analyze
    - start_date (datetime): Start date for the analysis
    - end_date (datetime): End date for the analysis
    - window_size (str): Size of the rolling window (e.g., '30D' for 30 days, '12W' for 12 weeks)
    - data_freq (str): Frequency of the data ('1D' for daily, '1W' for weekly)
    
    Returns:
    - tuple: (correlation_df, mean_correlation)
        - correlation_df: DataFrame with rolling correlations over time
        - mean_correlation: Average correlation over the entire period
    """
    if len(assets) != 2:
        raise ValueError("Exactly two assets must be provided for correlation analysis")
    
    # Fetch data for the assets
    data = yf.download(assets, start=start_date, end=end_date, interval=data_freq)['Close']
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Calculate rolling correlation
    rolling_correlation = returns[assets[0]].rolling(window=window_size).corr(returns[assets[1]])
    
    # Calculate mean correlation
    mean_correlation = returns[assets[0]].corr(returns[assets[1]])
    
    # Create correlation DataFrame
    correlation_df = pd.DataFrame({
        'Date': rolling_correlation.index,
        'Correlation': rolling_correlation.values
    })
    
    return correlation_df, mean_correlation

def run_correlation_analysis(assets, start_date, end_date, window_size='30D', data_freq='1D'):
    """
    Runs a complete correlation analysis for two assets and displays the results.
    
    Parameters:
    - assets (list): List of two asset tickers to analyze
    - start_date (datetime): Start date for the analysis
    - end_date (datetime): End date for the analysis
    - window_size (str): Size of the rolling window
    - data_freq (str): Frequency of the data
    """
    correlation_df, mean_correlation = analyze_asset_correlation(
        assets, start_date, end_date, window_size, data_freq
    )
    
    print(f"\nCorrelation Analysis Results for {assets[0]} and {assets[1]}:")
    print(f"Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Average Correlation: {mean_correlation:.3f}")
    
    plot_correlation_analysis(correlation_df, assets, mean_correlation, window_size)
    
    return correlation_df, mean_correlation

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate DCA for various scenarios.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--total_weekly_investment", type=float, default=None, help="Total weekly investment amount to override the scenario defaults.")
    parser.add_argument("--years", type=int, default=None, help="Number of years to override the scenario defaults.")
    return parser.parse_args()

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
    save_combined_metrics(all_detailed_metrics,"outputs")

    return all_summaries, all_detailed_metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCA Asset Simulator and Asset Correlation Analysis")
    
    # Original DCA simulation arguments
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--total_weekly_investment", type=float, default=None, help="Weekly investment override.")
    parser.add_argument("--years", type=int, default=None, help="Override for number of years.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save comparison plots.")
    
    # New correlation analysis arguments
    parser.add_argument("--correlation", action="store_true", help="Run correlation analysis")
    parser.add_argument("--asset1", type=str, help="First asset for correlation analysis")
    parser.add_argument("--asset2", type=str, help="Second asset for correlation analysis")
    parser.add_argument("--window", type=str, default="30D", 
                       help="Rolling window size for correlation (e.g., '30D', '12W', '3M')")
    parser.add_argument("--freq", type=str, default="1D",
                       help="Data frequency for correlation ('1D' for daily, '1W' for weekly)")
    
    args = parser.parse_args()

    # Run DCA simulation if config file is provided
    if args.config:
        dca_asset_simulator(
            config_path=args.config,
            total_weekly_investment=args.total_weekly_investment,
            years=args.years,
            output_dir=args.output_dir
        )
    
    # Run correlation analysis if requested
    if args.correlation:
        if not (args.asset1 and args.asset2):
            print("Error: Both --asset1 and --asset2 must be provided for correlation analysis")
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=(args.years or 5) * 365)
            
            print(f"\nRunning correlation analysis for {args.asset1} and {args.asset2}")
            print(f"Window size: {args.window}, Frequency: {args.freq}")
            
            correlation_df, mean_correlation = run_correlation_analysis(
                assets=[args.asset1, args.asset2],
                start_date=start_date,
                end_date=end_date,
                window_size=args.window,
                data_freq=args.freq
            )
            
            if args.output_dir:
                output_file = f"{args.output_dir}/correlation_{args.asset1}_{args.asset2}.csv"
                correlation_df.to_csv(output_file)
                print(f"Correlation data saved to: {output_file}")