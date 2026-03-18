from scipy.optimize import minimize
import numpy as np
#covar_matrix = returns_df.cov()*252
def risk_budget_objective(weights, covar_matrix):
  """
  Objective function for equal risk contribution optimization
  """
  portfolio_vol = np.sqrt(np.dot(weights, np.dot(covar_matrix, weights)))
  marginal_cont = np.dot(covar_matrix, weights)/portfolio_vol
  cont = weights*marginal_cont

  # Minimize sum of squared deviations from equal risk contribution
  target_cont = portfolio_vol/len(weights)
  return np.sum((cont - target_cont)**2)

def optimize_risk_parity(covar_matrix):
  """
  Optimize for equal risk contribution weights
  """
  n_assets = len(covar_matrix)

  # Initial guess
  x0 = np.ones(n_assets)/n_assets

  # Contraints: weights sum to 1
  constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
  bounds = [(0.001, 0.999) for _ in range(n_assets)]

  # Optimize
  result = minimize(risk_budget_objective,
                    x0,
                    args = (covar_matrix, ),
                    method = 'SLSQP',
                    bounds = bounds,
                    constraints = constraints,
                    options = {'ftol': 1e-12, 'disp': False})

  return result.x
