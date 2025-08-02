import pandas as pd

def validate_average_open(df):
    """
    Test if your average_open feature has predictive value
    """
    # Calculate average_open (assuming you have OPEN1, OPEN2, OPEN3)
    open_cols = [col for col in df.columns if col.startswith('OPEN')]
    if len(open_cols) >= 3:
        df['average_open'] = df[open_cols[:3]].mean(axis=1)
    else:
        print("Warning: Not enough OPEN columns found")
        return None
    
    # Test predictive power
    future_returns = df['truth'].shift(-1)
    prob_up = df['prob_up']
    
    # Test alignment signals
    bullish_alignment = (df['average_open'] > 1) & (prob_up > 0.5)
    bearish_alignment = (df['average_open'] < 1) & (prob_up < 0.5)
    
    results = {
        'bullish_alignment': {
            'avg_return': future_returns[bullish_alignment].mean(),
            'hit_rate': (future_returns[bullish_alignment] > 0).mean(),
            'count': bullish_alignment.sum()
        },
        'bearish_alignment': {
            'avg_return': future_returns[bearish_alignment].mean(),
            'hit_rate': (future_returns[bearish_alignment] < 0).mean(),  # Bearish success
            'count': bearish_alignment.sum()
        },
        'correlation': df['average_open'].corr(future_returns)
    }
    
    print("=== AVERAGE_OPEN VALIDATION RESULTS ===")
    print(f"Correlation with future returns: {results['correlation']:.4f}")
    print(f"Bullish alignment: Return={results['bullish_alignment']['avg_return']:.4f}, "
          f"Hit Rate={results['bullish_alignment']['hit_rate']:.2%}, "
          f"Count={results['bullish_alignment']['count']}")
    print(f"Bearish alignment: Return={results['bearish_alignment']['avg_return']:.4f}, "
          f"Hit Rate={results['bearish_alignment']['hit_rate']:.2%}, "
          f"Count={results['bearish_alignment']['count']}")
    
    return results



if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")

    # Usage
    open_results = validate_average_open(df)
