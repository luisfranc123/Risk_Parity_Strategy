def annual_weight_return(weights: dict, years: list, portfolio: str, returns_by_year: dict):
    
    """
    Function that calculates anuualized returns by year of a portfolio
        Args: 
         - weights: dictionary containing the asset weights of each portfolio asset 
         - years: list of years to evaluate
         - portfolio: str name of the portfolio. (e.g., asp, SPY_vol, SPY_bond, ETF)
         - returns_by_year: dictionary containing all returns per year per portfolio
    """
    results = {}
    for year in years:
        if year not in returns_by_year[portfolio]:
            print(f"Warning: {year} not found in Portfolio: {portfolio}, skipping.")
            continue
   
        df_year = returns_by_year[portfolio][year] # Daily return for each year
        actual_days = len(df_year)
        is_annualized = actual_days >= 200
        
        if is_annualized: # threshold for incomplete year
            per_asset_return = df_year.apply(lambda x: (1 + x).prod()**(252/len(x)) - 1) # Annualized return per asset for this year
            
        else:
            per_asset_return = (1 + df_year).prod() - 1 # cumulative only
            
        # Map weights and weighted sum 
        weighted_return = per_asset_return.index.map(weights)*per_asset_return
        results[year] = weighted_return.sum()*100

    return (results)