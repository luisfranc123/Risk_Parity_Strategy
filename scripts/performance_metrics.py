import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import statsmodels.api as sm

def calculate_performance_metrics(backtest_results, inflation_rate = 0.025, 
                                  risk_free_rate = 0.02, benchmark_returns = None):
    """
    calculate key performance statistics for a portfolio.

    Parameters:
    -----------
    backtest_results: pd.DataFrame
        DataFrame with 'date', 'portfolio_return', and 'portfolio_value' columns.
    inflation_rate: float
        Assumed annual inflation rate for calculating real returns.
    risk_free_rate: float
        Annual risk-free rate for Sharpe and Sortino ratios.
    benchmark_returns: pd.Series, optional
        Monthly returns of a benchmark (e.g., SPY) to calculate correlation.

    Returns:
    --------
    dict
        A dictionary containing all calculated performance metrics.
    """
    returns_series = backtest_results['portfolio_return']
    portfolio_values = backtest_results['portfolio_value']
    dates = backtest_results['date']
    initial_value = 10000
    
    # -- Align portfolio and benchmark --
    df1 = benchmark_returns.to_frame().reset_index()
    df2 = backtest_results[['date', 'portfolio_return']].rename(columns = {'date': 'Date'})
    aligned_df = pd.merge(df1, df2, on = 'Date', how = 'outer').dropna()
    bench_col = aligned_df.columns[1]
    port_col = aligned_df.columns[2]
    bench_aligned = aligned_df[bench_col]
    port_aligned = aligned_df[port_col]

    # Monthly risk-free rate
    monthly_rf = (1 + risk_free_rate)**(1/12) - 1
    
    # --- 1. Basic values ---
    #start_balance = portfolio_values.iloc[0]
    start_balance = initial_value
    end_balance = portfolio_values.iloc[-1]
    print(f"Start Balance: ${start_balance:,.2f}")
    print(f"End Balance: ${end_balance:,.2f}")

    # --- 2. years calculation ---
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days/365
    print(f"Period: {years:.2f} years ({dates.iloc[0].date()} to {dates.iloc[-1].date()})")
    
    # --- 3. Inflation-adjusted end balance ---
    end_balance_real = end_balance/((1 + inflation_rate)**years)
    print(f"End Balance (Inflation Adj., {inflation_rate:.1%} p.a.): ${end_balance_real:,.2f}")   

    # --- 4. Annualized Return (CAGR) ---
    total_return = (end_balance/start_balance) - 1
    cagr = (1 + total_return)**(1/years) - 1
    cagr_real = (1 + cagr)/(1 + inflation_rate) - 1
    print(f"Annualized Return (CAGR): {cagr:.2%}")
    print(f"Annualized Return (CAGR, Inflation Adj.): {cagr_real:.2%}")

    # --- 5. Annualized Volatility (Standard Deviation) ---
    annualized_vol = returns_series.std()*np.sqrt(12)
    print(f"Annualized Volatility (Std Dev): {annualized_vol:.2%}")

    # --- 6. Best and Worst Year ---
    # Group returns by year and calculate the total return for each year
    yearly_returns = (
        backtest_results.
            groupby(backtest_results['date'].dt.year)['portfolio_return']
            .apply(lambda x: (1 + x).prod() - 1)
    )
    if len(yearly_returns) > 0:
        best_year = yearly_returns.max()
        worst_year = yearly_returns.min()
        print(f"Best Year: {best_year:.2%}")
        print(f"Worst Year: {worst_year:.2%}")
    else:
        print("Best Year: N/A")
        print("Worst Year: N/A")

    # --- 7. Maximum Drawdown ---
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max)/rolling_max
    max_drawdown = drawdowns.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # --- 8. Sharpe Ratio ---
    # Annualized Sharpe ratio using the portfolio's returns and a risk-free rate
    # Assuming risk_free_rate is annual, convert to monthly for calculation
    excess_returns = returns_series - monthly_rf
    sharpe_ratio = (excess_returns.mean()*np.sqrt(12))/annualized_vol
    print(f"Sharpe Ratio (rf = {risk_free_rate:.1%}): {sharpe_ratio:.2f}")

    # --- 9. Sortino Ratio ---
    # Similar to Sharpe but penalizes only downside deviation
    downside_returns = returns_series[returns_series < monthly_rf]
    downside_deviation = downside_returns.std()*np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino_ratio = (excess_returns.mean()*np.sqrt(12))/downside_deviation if downside_deviation != 0 else np.nan
    print(f"Sortino Ratio (rf = {risk_free_rate:.1%}): {sortino_ratio:.2f}")

    # --- 10. Benchmark Correlation ---
    corr = bench_aligned.corr(port_aligned)
    print(f"Benchmark Correlation: {corr:.2f}")
    
    # --- 11. Arithmetic Mean (monthly) ---
    mean_monthly = returns_series.mean()
    print(f"Arithmetic Mean (monthly): {mean_monthly:.2%}")

    # --- 12. Arithmetic Mean (annualized) ---
    mean_annual = mean_monthly*12
    print(f"Arithmetic Mean (annualized): {mean_annual:.2%}")

    # --- 13. Geometric Mean (monthly) ---
    geom_mean_monthly = np.exp(np.mean(np.log(1 + returns_series))) - 1
    print(f"Geometric Mean (monthly): {geom_mean_monthly:.2%}")

    # --- 14. Geometric Mean (annualized) ---
    geom_mean_annual = (1 + geom_mean_monthly)**12 - 1
    print(f"Geometric Mean (annualized): {geom_mean_monthly:.2%}")

    # --- 15. Standard Deviation (monthly) ---
    std_monthly = returns_series.std()
    print(f"Standard Deviation (monthly): {std_monthly:.2%}")

    
    # --- 16. Standard Deviation (annualized) ---
    std_annual = returns_series.std()*np.sqrt(12)
    print(f"Standard Deviation (annualized): {std_annual:.2%}")

    # --- 17. Downside deviation (monthly) ---
    margin = 0.001 
    downside_deviations = np.clip([val - margin for val in returns_series], -np.inf, 0)

    # Calculate the mean of the squared downside deviations
    # We use np.nanmean to handle any potential NaNs in the input data.
    mean_squared_downside_deviations = np.nanmean(np.square(downside_deviations))

    # Downside deviation
    downside_dev = (np.sqrt(mean_squared_downside_deviations))*100
    print(f"Downside Deviation (monthly): {downside_dev:.2%}")

    # --- 18. OLS: Beta and R-Squared (State Street Global Allocation ETF as the benchmark) ---
    X_ols = sm.add_constant(bench_aligned)
    model = sm.OLS(port_aligned, X_ols).fit()
    beta = model.params.iloc[1]
    r_squared = model.rsquared
    print(f"Beta (State Street Global Allocation ETF as benchmark): {beta:.2f}")  
    print(f"R-Squared: {r_squared:.4f}")

    # --- 19. Alpha (annualized) ---
    # Define risk-free rate
    monthly_rf_series = (1 + risk_free_rate)**(1/12) - 1
    excess_port = port_aligned - monthly_rf_series
    excess_bench = bench_aligned - monthly_rf_series
    X_alpha = sm.add_constant(excess_bench)  
    model_alpha = sm.OLS(excess_port, X_alpha).fit()
    annual_alpha = model_alpha.params.iloc[0]*12
    print(f"Alpha (annualized): {annual_alpha:.4f}")

    # --- 20. Treynor Ratio (%) ---
    ann_excess_return = cagr - risk_free_rate
    treynor_ratio = (ann_excess_return/beta)*100 if beta != 0 else np.nan
    print(f"Treynor Ratio (%): {treynor_ratio:.2f}%")

    # --- 21. Modigliani-Modigliani (M2) measure ---
    # M2 = Sharpe_portfolio*Vol_benchmark + Rf
    # Tells you what the portfolio would return if it had the same risk as the benchmark
    bench_vol_ann = bench_aligned.std()*np.sqrt(12)
    m2_measure = (sharpe_ratio*bench_vol_ann) + risk_free_rate
    print(f"Modigliani-Modigliani (M²) Measure: {m2_measure:.2%}")

    # --- 22. Active Return ---
    # Annualized portfolio CAGR minus annualized benchmark CAGR
    bench_total_return = (1 + bench_aligned).prod() - 1
    bench_years = len(bench_aligned)/12
    bench_cagr = (1 + bench_total_return)**(1/bench_years) - 1
    active_return = cagr - bench_cagr
    print(f"Active Return (annualized): {active_return:.2%}")

    # --- 23. Tracking Error ---
    # annualized standard deviation of the active (excess) return series
    active_returns_series = port_aligned - bench_aligned
    tracking_error = active_returns_series.std()*np.sqrt(12)
    print(f"Tracking Error (annualized): {tracking_error:.2%}")

    # --- 24. Information ratio --- 
    information_ratio = (active_returns_series.mean()*12/tracking_error if tracking_error != 0 else np.nan)
    print(f"Information Ratio: {information_ratio:.2f}")

    # --- 25. Skewness ---
    skewness = returns_series.skew()
    print(f"Skewness: {skewness:.4f}")

    # -- 26. Excess Kurtosis ---
    excess_kurtosis = returns_series.kurt()
    print(f"Excess Kurtosis: {excess_kurtosis:.4f}")

    # --- 27. Historical VaR (5%) ---
    # The 5th percentile of the empirical return distribution 
    hist_var_5 = np.percentile(returns_series, 5)
    print(f"Historical VaR (5%): {hist_var_5:.2%}")

    # --- 28. Analytical VaR (5%) ---
    # Parametric (Gaussian) VaR: mean - 1.645*std
    z_95 = stats.norm.ppf(0.05)
    analytical_var_5 = mean_monthly + z_95*std_monthly
    print(f"Analytical VaR (5%):  {analytical_var_5:.2%}")

    # --- 29. Conditional VaR / Expected Shortfall (5%) ---
    # Average of returns that fall below the Historical VaR threshold
    cvar_5 = returns_series[returns_series <= hist_var_5].mean()
    print(f"Conditional VaR (5%):  {cvar_5:.2%}")

    # --- 30. Upside Capture Ratio (%) ---
    # How much of the benchmark's UP months the portfolio captures
    up_mask = bench_aligned > 0
    up_port_return = (1 + port_aligned[up_mask]).prod() - 1
    up_bench_return = (1 + bench_aligned[up_mask]).prod() - 1
    up_port_ann = (1 + up_port_return)**(12/up_mask.sum()) - 1
    up_bench_ann = (1 + up_bench_return)**(12 / up_mask.sum()) - 1
    upside_capture = (up_port_ann/up_bench_ann)*100 if up_bench_ann != 0 else np.nan
    print(f"Upside Capture Ratio (%): {upside_capture:.2f}%")

     # -- 31. Downside Capture Ratio (%) ---
    # How much of the benchmark's DOWN months the portfolio captures
    down_mask = bench_aligned < 0
    down_port_return = (1 + port_aligned[down_mask]).prod() - 1
    down_bench_return = (1 + bench_aligned[down_mask]).prod() - 1
    down_port_ann = (1 + down_port_return)**(12/down_mask.sum()) - 1
    down_bench_ann = (1 + down_bench_return)**(12/down_mask.sum()) - 1
    downside_capture = (down_port_ann/down_bench_ann)*100 if down_bench_ann != 0 else np.nan
    print(f"Downside Capture Ratio (%): {downside_capture:.2f}%")

    # --- 32. Positive Periods ---
    n_positive = (returns_series > 0).sum()
    n_total = len(returns_series)
    pct_positive = n_positive/n_total
    print(f"Positive Periods: {n_positive}/{n_total} ({pct_positive:.2%})")

    # --- 33. Gain/Loss Ratio ---
    # Average gain of positive months divided by average absolute loss of negative months
    avg_gain = returns_series[returns_series > 0].mean()
    avg_loss = returns_series[returns_series < 0].mean()
    gain_loss_ratio = (avg_gain/abs(avg_loss)) if avg_loss != 0 else np.nan
    print(f"Gain/Loss Ratio: {gain_loss_ratio:.2f}")
  
    return {
        # ── Balances ──
        'Start_Balance': start_balance,
        'End_Balance': end_balance,
        'End_Balance_Real': end_balance_real,
        # ── Returns ──
        'CAGR': cagr,
        'CAGR_Real': cagr_real,
        'Arithmetic_Mean_Monthly': mean_monthly,
        'Arithmetic_Mean_Annual': mean_annual,
        'Geometric_Mean_Monthly': geom_mean_monthly,
        'Geometric_Mean_Annual': geom_mean_annual,
        # ── Risk ──
        'Annualized_Volatility': annualized_vol,
        'Std_Dev_Monthly': std_monthly,
        'Std_Dev_Annual': std_annual,
        'Downside_Deviation_Monthly': downside_dev,
        'Maximum_Drawdown': max_drawdown,
        'Best_Year': best_year,
        'Worst_Year': worst_year,
        # ── Distribution ──
        'Skewness': skewness,
        'Excess_Kurtosis': excess_kurtosis,
        # ── Value-at-Risk ──
        'Historical_Value-at-Risk_(5%)': hist_var_5,
        'Analytical_Value-at-Risk_(5%)': analytical_var_5,
        'Conditional_Value-at-Risk_(5%)': cvar_5,
        # ── Risk-Adjusted Returns ──
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Treynor_Ratio_Pct': treynor_ratio,
        'Modigliani_M2_Measure': m2_measure,
        # ── Regression/Factor ──
        'Alpha_Annualized': annual_alpha,
        'Beta': beta,
        'R_Squared': r_squared,
        # ── Benchmark-Relative ──
        'Benchmark_Correlation': corr,
        'Active_Return': active_return,
        'Tracking_Error': tracking_error,
        'Information_Ratio': information_ratio,
        'Upside_Capture_Ratio_(%)': upside_capture,
        'Downside_Capture_Ratio_(%)': downside_capture,
        # ── Win/Loss ──
        'Positive_Periods': n_positive,
        'Total_Periods': n_total,
        'Positive_Periods (%)': pct_positive,
        'Gain_Loss_Ratio': gain_loss_ratio,
    }
