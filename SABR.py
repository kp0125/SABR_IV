import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysabr import Hagan2002LognormalSABR
from pysabr import hagan_2002_lognormal_sabr as sabr
from pysabr.black import lognormal_call
import time
import datetime as dt



# construct the options chain 
def option_chains(ticker):
    """
    """
    asset = yf.Ticker(ticker)
    expirations = asset.options
    
    last_price = asset.history()['Close'][-1]
    
    chains = pd.DataFrame()
    
    for expiration in expirations:
        # tuple of two dataframes
        opt = asset.option_chain(expiration)
        
        calls = opt.calls
        calls['OptionType'] = "call"
        
        puts = opt.puts
        puts['OptionType'] = "put"
        
        chain = pd.concat([calls, puts])
        chain['Expiration'] = pd.to_datetime(expiration, utc=True)
        
        chains = pd.concat([chains, chain])
    
    chains['UnderlyingSymbol'] = ticker
    chains['UnderlyingPrice'] = last_price
    
    return chains


# define a Options object and ticker
options_chain = option_chains("SPY")


# Organize the columns 
columns = {
    "contractSymbol": "OptionSymbol",
    "volume": "Volume",
    "openInterest": "OpenInterest",
    "lastTradeDate": "QuoteDatetime",
    "change": "OptionChange",
    "lastPrice": "Last",
    "bid": "Bid",
    "ask": "Ask",
    "strike": "Strike",
    "impliedVolatility": "IV",
    "percentChange": "PctChg"

}

options_frame.rename(columns=columns, inplace=True)


def _get_days_until_expiration(series):

    expiration = series["Expiration"]

    # add the hours to the expiration date 
    date_str = expiration.strftime("%Y-%m-%d") + " 23:59:59"

    # convert date string 
    expiry = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    # get the date for today
    today = dt.datetime.today()

    # return the difference and add one 
    return (expiry - today).days + 1



def _get_time_fraction_until_expiration(series):

    expiration = series["Expiration"]

    # add the hours to the expiration date 
    date_str = expiration.strftime("%Y-%m-%d") + " 23:59:59"

    # convert date string 
    time_tuple = time.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    # get the number of seconds until expiration
    expiry_in_seconds_from_epoch = time.mktime(time_tuple)

    # get the number of seconds to right now
    today = dt.datetime.today()
    right_now_in_seconds_from_epoch = time.time()

    # get the total number of seconds to expiration
    seconds_until_expiration = (
        expiry_in_seconds_from_epoch - right_now_in_seconds_from_epoch
    )

    # seconds in year
    seconds_in_year = 31536000.0

    # fraction of seconds to expiration to total in year
    return max(seconds_until_expiration / seconds_in_year, 1e-10)


def _get_mid(series):
    
    bid = series["Bid"]
    ask = series["Ask"]
    last = series["Last"]

    # if the bid or ask doesn't exist, return 0.0
    if np.isnan(ask) or np.isnan(bid):
        return 0.0

    # if the bid or ask are 0.0, return the last traded price
    elif ask == 0.0 or bid == 0.0:
        return last
    else:
        return (ask + bid) / 2.0


options_frame["DaysUntilExpiration"] = options_frame.apply(
    _get_days_until_expiration, axis=1
)
options_frame["TimeUntilExpiration"] = options_frame.apply(
    _get_time_fraction_until_expiration, axis=1
)

options_frame["Mid"] = options_frame.apply(_get_mid, axis=1)

# select an expiration date
Expiration = "2026-01-16"

calls = options_frame[options_frame.OptionType == "call"]
jan_2026_c = calls[calls.Expiration == Expiration].set_index("Strike")
jan_2026_c["Mid"] = (jan_2026_c.Bid + jan_2026_c.Ask) / 2

puts = options_frame[options_frame.OptionType == "put"]
jan_2026_p = puts[puts.Expiration == Expiration].set_index("Strike")
jan_2026_p["Mid"] = (jan_2026_p.Bid + jan_2026_p.Ask) / 2

strikes = jan_2026_c.index
vols = jan_2026_c.IV * 100

# fit the SABR model, use put/call parity
f = (
    (jan_2026_c.Mid - jan_2026_p.Mid)
    .dropna()
    .abs()
    .sort_values()
    .index[0]
)
t = (pd.Timestamp(Expiration) - pd.Timestamp.now()).days / 365
beta = 0.5


sabr_lognormal = Hagan2002LognormalSABR(
    f=f,
    t=t,
    beta=beta
)

alpha, rho, volvol = sabr_lognormal.fit(strikes, vols)

calibrated_vols = [
    sabr.lognormal_vol(strike, f, t, alpha, beta, rho, volvol) * 100
    for strike in strikes
]

plt.plot(
    strikes, 
    calibrated_vols
)

plt.xlabel("Strike")
plt.ylabel("Volatility")
plt.title("Volatility Smile")
plt.plot(strikes, vols)
plt.show()
