from optimize_portfolio import * 
import numpy as np
import pandas as pd
from datetime import datetime

def load_dividend_returns(div_csv_path, prices):
    """
    load a dividend CSV and convert dollar-per-share amounts into 
    daily return contributions (dividend/price on ex-date).

    Parameters: 
    -----------
    - div_csv_path: str
        Path to a dividens CSV file (dates*assets, values in $/share).
    - prices: pd.DataFrame
        Daily price (or NAV) dataframe with the same asset columns. 
        Used to convert dollar dividends into return equivalents.

    Returns:
    -----------
    pd.DataFrame
        Daily dividend return contributions (same shape as prices, 
        but only for the assets present in the CSV). Missing dates are filled with 0. 
    """
    divs = pd.read_csv(div_csv_path, index_col = 0, parse_dates = True)

    div_returns = pd.DataFrame(0.0, index = prices.index, columns = divs.columns)

    for asset in divs.columns:
        if asset not in prices.columns:
            continue

        # Only process dates where a dividend was actually paid
        pay_dates = divs.index[divs[asset] != 0]

        for date in pay_dates:
            if date not in prices.index:
                # Find the closest prior trading date
                prior = prices.index[prices.index <= date]
                if prior.empty:
                    continue
                price_date = prior[-1]
            else:
                price_date = date

            price = prices.loc[price_date, asset]

            if pd.isna(price) or price == 0: 
                continue

            div_return = divs.loc[date, asset]/price

            # Assign the dividend return to the correct date in the index
            if date in div_returns.index:
                div_returns.loc[date, asset] += div_return
            else:
                # Use closest prior trading date if ex-date is not a trading day
                if price_date in div_returns.index:
                    div_returns.loc[price_date, asset] += div_return
                

    return div_returns    

def walk_forward_backtest(returns, lookback_months = 0, 
                          initial_value = 10000, 
                          rebalance_frequency = 'monthly', 
                          dividend_yield = 0.0, 
                          div_csv_path = None, 
                          prices = None):
    """
    Perform a walk-forward backtest of the risk parity strategy.

    Parameters:
    -----------
    - returns: pd.DataFrame
        Daily returns dataframe with asset columns
    - lookback_months: int
        Number of months to use for covariance estimation
    - initial_value: float
        Starting portfolio value
    - rebalance_frequency: str
        'monthly' or 'quarterly' — how often to reoptimize weights
    - dividend_yield: float or dict
        Annual dividend yield(s) to reinvest. Use a single float (e.g. 0.02)
        for a uniform yield across all assets, or dict mapping asset names 
        to their individual annual yields (e.g. {'SPY': 0.013, 'TLT': 0.035})
    - div_csv_path: str or None
        Path to a portfolio-specific dividends CSV (dates * assets, 
        values in $/share). When provided the CSV dividends are 
        converted to daily return contributions and aggregated to 
        monthly figures; the dividend_yield parameter is then ignored. 
    - prices: pd.DataFrame or None
        Daily price (NAV) dataframe required to convert $/share
        dividends into returns. Must be supplied when div_csv_path
        is given. 
        
    Returns:
    -----------
    pd.DataFrame
        Backtest results with dates, returns, values, and weights
    """
    results = []
    portfolio_values = [initial_value]

    # Convert returns to monthly frequency for rebalancing
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # ---------- #
    # Dividend handling

    n_assets = returns.shape[1]
 
    if div_csv_path is not None:
        # --- Path A: use actual per-share dividends from CSV --
        if prices is None:
            raise ValueError(
                "A 'prices' DataFame must be supplied together with "
                "'div_csv_path' to convert $/share dividends to returns."
            )

        # Build a daily dividend-return DataFrame aligned to 'returns'
        daily_div_returns = load_dividend_returns(div_csv_path, prices)

        # Align columns to the portfolio-s asset data (fill 0 for 
        # assets not present in the dividend file)
        daily_div_returns = daily_div_returns.reindex(
            columns = returns.columns, fill_value = 0.0
        )

        # Aggregate to monthly dividend return contributions 
        # (compound within each month, consistent with price returns)
        monthly_div_returns = daily_div_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # Reindex to match monthly_returns (fill missing months with 0)
        monthly_div_returns = monthly_div_returns.reindex(
            monthly_returns.index, fill_value = 0.0
        )

        use_csv_divs = True

    else:
        # --- Path B: fallback - uniform annual yield converted to monthly ---
        if isinstance(dividend_yield, dict):
            monthly_div = np.array([
                (1 + dividend_yield.get(col, 0.0))**(1/12) - 1
                for col in returns.columns
            ])
        else:
            monthly_div = np.full(n_assets, (1 + dividend_yield)**(1/12) - 1) 

        use_csv_divs = False     

    ## -- Walk forward Loop --#

    # -- Rebalancing: track current weights between rebalance periods --
    current_weights = None
    last_rebalance_month = None
    
    for i in range(lookback_months, len(monthly_returns)):

        current_date = monthly_returns.index[i]

        # 1. Decide whether to rebalance this month
        if rebalance_frequency == 'quarterly':
            # Rebalance at (Jan/Apr/Jul/Oct)
            is_rebalance_month = current_date.month in [3, 6, 9, 12]
        else: # default: monthly
            is_rebalance_month = True

        # Always optimize on the first iteration, regardless of rebalance schedule
        is_first = (current_weights is None)
        
        if is_first or is_rebalance_month:
            # 2. Build training data up to (and including) current month
            train_data = returns.loc[:current_date]
            train_monthly = train_data.resample('ME').apply(lambda x: (1 + x).prod() - 1)

            # 3. Calculate covariance matrix from training data (annualized)
            train_covar = train_monthly.cov()*12
            
            # 4. Optimize risk-parity weights
            current_weights = optimize_risk_parity(train_covar.values)

        # 5. Apply weights to current month's asset returns
        current_month_return = monthly_returns.iloc[i].values

        # 6. Add dividend income (weighted by current allocation) and reinvest
        if use_csv_divs:
            # Per asset dividend returns for this month ($/share -> return)
            monthly_div_array = monthly_div_returns.loc[current_date].values
            div_income = np.dot(current_weights, monthly_div_array)
        else:
            # Fallback: uniform annual-yield approximation
            div_income = np.dot(current_weights, monthly_div)
        
        portfolio_return = np.dot(current_weights, current_month_return) + div_income

        # 7. Update portfolio value 
        new_value = portfolio_values[-1]*(1 + portfolio_return)
        portfolio_values.append(new_value)


        results.append({
            'date': current_date,
            'portfolio_return': portfolio_return,
            'portfolio_value': new_value,
            'weights': dict(zip(returns.columns, current_weights))
        })
    

    return pd.DataFrame(results)

