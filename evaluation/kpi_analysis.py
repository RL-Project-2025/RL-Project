import numpy as np


def analyse_kpis(a2c_returns, random_returns, heuristic_returns=None):
    """
    Compare episode-level KPIs across policies.
    """

    print("\n===== KPI ANALYSIS =====")

    def summary(name, returns):
        returns = np.array(returns)
        print(f"\n{name}:")
        print(f"  Mean return: {returns.mean():.2f}")
        print(f"  Std return:  {returns.std():.2f}")
        print(f"  Median:     {np.median(returns):.2f}")
        print(f"  5th pct:    {np.percentile(returns, 5):.2f}")
        print(f"  95th pct:   {np.percentile(returns, 95):.2f}")

    summary("A2C", a2c_returns)
    summary("Random", random_returns)

    if heuristic_returns is not None:
        summary("Heuristic", heuristic_returns)
