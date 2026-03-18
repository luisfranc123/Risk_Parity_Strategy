# # Volatility trigger check
import numpy as np
def volatility_trigger_check(returns, lookback_days = 60, threshold = 0.20):
  """
  Check if any asset's volatility has changed significantly.
  """

  recent_returns = returns.tail(lookback_days)
  historical_returns = returns.iloc[:-lookback_days]

  recent_vol = recent_returns.std()*np.sqrt(252)
  historical_vol = historical_returns.std()*np.sqrt(252)

  vol_change = abs(recent_vol - historical_vol)/historical_vol
  trigger_assets = vol_change[vol_change > threshold]

  if len(trigger_assets) > 0:
    print("Volatility trigger activated for:\n")
    for asset, change in trigger_assets.items():
      print(f"{asset}: {change:.1%} volatility change")
    return True
  return False
