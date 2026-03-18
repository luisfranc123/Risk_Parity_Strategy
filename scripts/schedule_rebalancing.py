from datetime import datetime
from dateutil.relativedelta import relativedelta

def should_rebalance(last_rebalance_date, frequency = 'monthly'):
  """
  Check if the portfolio needs rebalancing based on the given schedule.
  Supports 'monthly' and 'quarterly' frequencies.
  """

  today = datetime.now()
  if frequency == 'monthly':
      next_rebalance = last_rebalance_date + relativedelta(months = 1)
  elif frequency == 'quarterly':
      next_rebalance = last_rebalance_date + relativedelta(months = 3)
  else:
      raise ValueError(f"Unsupported frequency {frequency}. Use 'monthly' or 'quarterly'.")

  return today >= next_rebalance

def calculate_rebalancing_trades(current_weights, target_weights, portfolio_value):
  """
  Calculate the trades required to reach the target allocation.
  """
  weight_diff = target_weights - current_weights
  dollar_trades = weight_diff*portfolio_value

  # Filter out small trades (less than $100)
  significant_trades = dollar_trades[abs(dollar_trades) > 100]
  return significant_trades
