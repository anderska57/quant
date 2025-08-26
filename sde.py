"""
sde

Desc: Library of common stochastic differential equations 

@author: Katherine Anderson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

    
def gwp(T=10, n_paths=100, a=0.1, b=0.1, steps_per_year=12, x0=0):
    """
    generalized wiener process
    dx = adt + bdz, dz = eps*\sqrt(dt)

    Parameters
    ----------
    T : positive scalar (float)
        time in years
    n_paths : positive integer
        number of paths (scenarios)
    a : postive scalar (float)
        drift rate
    b : positive scalar (float)
        standard deviation/volatility
        b^2 = variance rate
    steps_per_year : positive integer
        number of steps per year
    x0 : scalar (float)
        initial value of x    

    Returns
    -------
    x_df : dataframe, size (n_steps+1,n_paths)
        x values with each column representing one path
        Note: n_steps is the number of time steps from x0 to T

    """
    
    # Define step size and number of steps per path
    dt = 1/steps_per_year
    n_steps = T*steps_per_year
    
    # Simulate epsilon values: array of standard normal random variables
    # size: (n_steps,n_paths)
    eps = np.random.normal(loc=0, scale=1, size=(n_steps,n_paths))
    
    # Calculate x-values at each time step for each path
    dx = a*dt*np.ones((n_steps,n_paths)) + b*np.sqrt(dt)*eps
    x = np.cumsum(dx, axis=0)
    
    # Concatenate time 0 values
    x = np.vstack((x0*np.ones((1,n_paths)),x))
    
    # Put data into dataframe with times as index
    times = dt*np.arange(0,n_steps+1) 
    x_df = pd.DataFrame(x, index=times, columns=np.arange(1,n_paths+1))
    
    return x_df


def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate

    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to annual

    """
    return np.log1p(r)


def gbm(n_years=10, n_scenarios=100, mu=0.07, sigma=0.15, steps_per_year=12, 
        s0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for stock 
    prices through Monte Carlo
    GBM equation: dS = mu*S*dt + sigma*S*dZ
    

    Parameters
    ----------
    n_years : positive integer, optional
        DESCRIPTION. Number of years 
    n_scenarios : positive integer, optional
        DESCRIPTION. number of scenarios or paths
    mu : positive scalar (float), optional
        DESCRIPTION. expected annualized return
    sigma : positive scalar (float), optional
        DESCRIPTION. annualized volatility
    steps_per_year : positive integer, optional
        DESCRIPTION. number of time steps per year
    s0 : scalar (float), optional
        DESCRIPTION. initial S value
    prices : Boolean, optional
        DESCRIPTION. True to return prices. False to return rates.

    Returns
    -------
    s_df : dataframe, size (n_steps+1, n_scenarios)
        DESCRIPTION: S-values, where each column represents one path of GBM

    """
    
    # Derive stepsize (dt) and number of steps
    dt = 1/steps_per_year
    n_steps = int(steps_per_year*n_years) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)),
                                   size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
    
    return ret_val


def show_gbm(n_scenarios, mu, sigma):
    """
    Draw results of stock price evolution under geometric brownian motion

    """
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma)
    fig, (x_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, 
                                        gridspec_kw={'width_ratios':[3,2]},
                                        figsize=(24,9))
    plt.subplots_adjust(wspace=0.0)
    
    prices.plot(ax=x_ax, legend=False)
    
    # Determine deterministic line (sigma=0)
    p_deterministic = gbm(n_scenarios=1, mu=mu, sigma=0)
    p_deterministic.plot(ax=x_ax, color='yellow', linewidth=3, legend=False)
    x_ax.set_xlabel('Time (Months)', fontsize=16)
    x_ax.set_title('Simulation of Geometric Brownian Motion', fontsize=20)
    hist_ax.set_title('Distribution of Realized Values at time T', fontsize=20)
    
    prices.iloc[-1,:].plot.hist(ax=hist_ax, bins=50, orientation='horizontal')
    hist_ax.set_xlabel('frequency', fontsize=16)
    
    return plt


def vasicek(n_years=10, n_scenarios=1, k=0.05, theta=0.03, sigma=0.05, 
            steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    returned values are annualized rates as well

    """
    if r_0 is None: 
        r_0 = sigma
        
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## price generation
    prices = np.empty_like(shock)
    
    def price(ttm,r): # ttm = time to maturity
        B = (1-np.exp(-k*ttm))/k
        A = np.exp(((B-ttm)*(k**2*theta-sigma**2/2))/(k**2) - ((sigma**2*B**2)/(4*k)))
        P = A*np.exp(-B*r)
        return P

    prices[0] = price(n_years, r_0)

    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = k*(theta-r_t)*dt + sigma*shock[step]
        rates[step] = r_t + d_r_t
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))

    return rates, prices          
    
    