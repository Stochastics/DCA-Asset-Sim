
# Dollar-Cost Averaging (DCA) Simulation Program

This program simulates the performance of dollar-cost averaging (DCA) investments into multiple assets (e.g., BTC, QQQ, VTI/SPY) over the past `N` years. It calculates annualized returns based on a fixed weekly investment amount and supports scenario-based configurations.

## Features

- Fetches historical price data using `yfinance`.
- Simulates DCA investments on a weekly basis.
- Supports scenario-based configurations defined in a YAML file.
- Calculates total returns and annualized returns for each asset.
- Provides portfolio-level metrics including total and annualized returns.
- Visualizes asset prices over time.

---

## Requirements

- Python 3.6 or higher
- Pip (Python package installer)
- Internet connection (to fetch historical price data)

---

## Setup Instructions

### 1. Clone or Download the Repository

```bash
git clone git@github.com:Stochastics/DCA-Asset-Sim.git
cd DCA-Asset-Sim
```
---

### 2. Set Up the Virtual Environment

# Create a virtual environment and load packages sh script
 ./setup.sh
source dcaenv/bin/activate

**Note**: The `requirements.txt` file is where new dependencies can be added. The shell script pulls this script from the repo.

---

### 3. Configure Scenarios

The simulation uses a YAML configuration file named `scenarios.yaml`. Define your scenarios in this file, specifying the assets, weights, and scenario name.

#### Example `scenarios.yaml`:
```yaml
scenarios:
  - name: "Balanced Portfolio"
    assets: ["BTC-USD", "QQQ", "VTI"]
    weights: [0.4, 0.4, 0.2]

  - name: "High Risk"
    assets: ["BTC-USD", "QQQ"]
    weights: [0.8, 0.2]

  - name: "Conservative"
    assets: ["QQQ", "VTI"]
    weights: [0.5, 0.5]
```

- **`name`**: The scenario's label for reporting.
- **`assets`**: List of asset tickers to simulate.
- **`weights`**: Allocation percentages for each asset. These should sum to 1.0 or less.

---

### 4. Kick Off the Simulation

Run the simulation program to process the scenarios defined in `scenarios.yaml`:

```bash
python DCA_SIMULATOR.py
```

The program will output:
1. Detailed results for each asset in each scenario.
2. Portfolio-level total returns and annualized returns.
3. A summary of results for all scenarios.

---

## Example Output

```plaintext
Running scenario: Balanced Portfolio
         Total Investment   Final Value  Total Return (%)  Annualized Return (%)
BTC-USD             523.0  33935.291070       6388.583379              51.780154
QQQ                 784.5   2206.747660        181.293519              10.896023
VTI                1307.5   2724.984648        108.411828               7.619814
Total              2615.0  38867.023378       5595.847255              30.981801
Scenario Balanced Portfolio completed.
```

---
## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

stochastics
 
Feel free to reach out for suggestions or improvements!


