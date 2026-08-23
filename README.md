# Black-Scholes-Pricing
convert_arb_pricer.py
=====================

Black–Scholes and Merton pricing utilities for convertible-arbitrage desks.
• Bloomberg download via pdblp (override with your own fetcher)
• Vectorised over pandas Series / DataFrame
• Designed for quick integration into Jupyter notebooks
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 1. Data helper – thin wrapper around Bloomberg's pdblp
# ---------------------------------------------------------------------------
def get_bbg_data(tickers: list[str],
                 fields: list[str],
                 start_date: str = None,
                 end_date: str = None,
                 periodicity: str = "DAILY") -> pd.DataFrame:
    """
    Pull historical data from Bloomberg.
    • tickers   : list like ["IBM US Equity", "AAPL US Equity"]
    • fields    : list like ["PX_LAST", "IVOL_MID", "YAS_SPREAD"]
    • start/end : YYYY-MM-DD strings
    Returns a multi-index DataFrame (date × ticker).
    """
    try:
        import pdblp
    except ImportError:
        raise ImportError("Install pdblp or replace get_bbg_data with your own fetcher")

    con = pdblp.BCon(timeout=20000)  # 20 s socket timeout
    con.start()
    df = con.bdh(tickers, fields, start_date, end_date, Per=periodicity)
    df = df.stack(level=0).rename_axis(["date", "ticker"])
    df = df.reset_index().set_index(["ticker", "date"]).sort_index()
    return df

# ---------------------------------------------------------------------------
# 2. Black–Scholes (risk-neutral, no dividends)
# ---------------------------------------------------------------------------
def black_scholes_price(S: float,
                        K: float,
                        T: float,
                        vol: float,
                        r: float = 0.0,
                        q: float = 0.0,
                        option_type: str = "call") -> float:
    """
    Vanilla European option (Black–Scholes-Merton).
    S  : spot
    K  : strike
    T  : time-to-expiry in years
    vol: annualised σ (implied volatility)
    r  : risk-free rate (continuously comp.)
    q  : dividend yield (continuously comp.)
    """
    if T <= 0 or vol <= 0:
        raise ValueError("T and vol must be positive")

    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type.lower() == "call":
        price = (np.exp(-q * T) * S * norm.cdf(d1) -
                 np.exp(-r * T) * K * norm.cdf(d2))
    elif option_type.lower() == "put":
        price = (np.exp(-r * T) * K * norm.cdf(-d2) -
                 np.exp(-q * T) * S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# ---------------------------------------------------------------------------
# 3. Merton structural model for credit-adjusted option value
# ---------------------------------------------------------------------------
def merton_call_price(S: float,
                      K: float,
                      T: float,
                      vol_asset: float,
                      r: float,
                      credit_spread: float) -> float:
    """
    Merton (1974) equity/option value as a levered call on assets.
    • S              : current asset value (proxy = EV or convert parity)[11:45 AM]• K              : debt face value (default barrier)
    • T              : time horizon in years
    • vol_asset      : volatility of assets (not equity IV)
    • r              : risk-free rate
    • credit_spread  : current credit spread of the issuer (decimal, not bp)
    Steps:
      – discount rate = r + credit_spread
      – risk-neutral default barrier = K
      – equity = call(S, K, T, vol_asset, r, 0)
      – Debt value = PV * N(-d2) + [some optional recovery term]
    Returning just the equity call price (convertible option component).
    """
    effective_r = r + credit_spread         # spread-adjusted discount
    return black_scholes_price(S, K, T, vol_asset, effective_r, 0.0, "call")

# ---------------------------------------------------------------------------
# 4. Convenience wrapper for vectorised usage with DataFrame rows
# ---------------------------------------------------------------------------
def bs_merton_wrapper(row: pd.Series) -> pd.Series:
    """
    Row-wise application: expects indexes
      ['spot', 'strike', 'ttm', 'iv', 'rf', 'div_yield', 'credit_spread']
    Returns both BS and Merton values.
    """
    bs_px = black_scholes_price(row.spot, row.strike, row.ttm,
                                row.iv, row.rf, row.div_yield, "call")
    merton_px = merton_call_price(row.spot, row.strike, row.ttm,
                                  row.iv, row.rf, row.credit_spread)
    return pd.Series({"bs_price": bs_px, "merton_price": merton_px})

# ---------------------------------------------------------------------------
# Example usage – run this block in a notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. Pull IBM spot + IV + 5y CDS spread ---
    # df = get_bbg_data(["IBM US Equity", "IBM5Y Corp"],  # example tickers
    #                   ["PX_LAST", "IVOL_MID", "LAST_PRICE"],
    #                   start_date="2026-08-16", end_date="2026-08-23")
    # (For demo below we’ll hard-code values.)
    data = {
        "spot": 180.25,           # IBM spot
        "strike": 185,
        "ttm": 0.5,               # 6 months
        "iv": 0.28,               # 28 % implied vol
        "rf": 0.04,               # 4 % risk-free
        "div_yield": 0.015,       # 1.5 % dividend
        "credit_spread": 0.012    # 120 bp CDS
    }
    row = pd.Series(data)
    res = bs_merton_wrapper(row)
    print(res)
