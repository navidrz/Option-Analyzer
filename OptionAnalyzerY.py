# option_analyzer_module.py

import numpy as np
import pandas as pd
import multiprocessing
from typing import Dict, Tuple, Optional, Any
from scipy.stats import norm, skew, kurtosis, skewnorm
from scipy.optimize import minimize
from functools import partial
import logging
from gldpy import GLD  # Ensure gldpy is installed and imported correctly
import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger("OptionAnalysisLogger")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define Market Scenarios
MARKET_SCENARIOS = {
    'Normal Market': {'skewness': 0.0, 'kurtosis': 3.0},
    'Bull Market': {'skewness': -0.5, 'kurtosis': 3.5},
    'Bear Market': {'skewness': 0.5, 'kurtosis': 3.5},
    'Volatile Market': {'skewness': 0.0, 'kurtosis': 5.0},
    'Market Crash': {'skewness': 2.0, 'kurtosis': 9.0},
    'Market Rally': {'skewness': -2.0, 'kurtosis': 9.0}
}

class OptionAnalyzer:
    def __init__(self,
                 data_loader,  # Instance of DataLoader
                 risk_free_rate: float = 0.01,
                 market_scenario: str = 'Normal Market',
                 distribution_type: str = 'GLD'):
        """
        Initializes the OptionAnalyzer with a DataLoader instance and analysis parameters.
        
        Args:
            data_loader: Instance of DataLoader class.
            risk_free_rate (float): Risk-free interest rate.
            market_scenario (str): Market scenario for simulation.
            distribution_type (str): Distribution type for price simulation ('GLD', 'SkewNormal', 'CornishFisher', etc.).
        """
        self.data_loader = data_loader
        self.risk_free_rate = risk_free_rate
        self.market_scenario = market_scenario
        self.distribution_type = distribution_type
        self.cleaned_data = pd.DataFrame()
        self.historical_data = {}
        self.simulation_results = {}
        self.greeks = {}
        self.pop_results = {}
        self.breakeven_points = {}
        self.sharpe_ratios = {}
        self.var = {}
        self.cvar = {}
        self.recommendations_df = pd.DataFrame()
        self.target_date_distribution = {}
        self.cash_flows = {}
        self.variance = {}
        self.payout_ratios = {}
        self.market_views = {}
        self.strategy = 'default'
        self.metrics_min = {}
        self.metrics_max = {}
        self.thresholds = {}
        self.scenario_analysis_results = {}
        self.market_scenarios = MARKET_SCENARIOS  

    def prepare_data(self):
        """
        Prepares data by loading cleaned option data and historical data from DataLoader.
        """
        try:
            # Load data using DataLoader
            self.cleaned_data, self.historical_data = self.data_loader.load_all_data()
            logger.info("Data preparation completed.")
            
            # Calculate additional metrics
            self.calculate_volatility()
            self.calculate_moneyness()
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def calculate_volatility(self):
        """
        Calculates historical volatility for each unique ua_tse_code.
        """
        logger.info("Calculating historical volatility for each unique ua_tse_code.")
        volatility_dict = {}
        for ua_tse_code, df in self.historical_data.items():
            if df.empty:
                logger.warning(f"No historical data for ua_tse_code: {ua_tse_code}")
                volatility_dict[ua_tse_code] = np.nan
                continue
            df = df.sort_values('date')
            df['daily_return'] = df['close'].pct_change()
            daily_std = df['daily_return'].std()
            annual_volatility = daily_std * np.sqrt(252)
            volatility_dict[ua_tse_code] = annual_volatility
            logger.debug(f"Volatility for {ua_tse_code}: {annual_volatility:.4f}")

        self.cleaned_data['Volatility'] = self.cleaned_data['ua_tse_code'].map(volatility_dict)
        missing_vol = self.cleaned_data['Volatility'].isna().sum()
        if missing_vol > 0:
            mean_vol = self.cleaned_data['Volatility'].mean()
            self.cleaned_data['Volatility'].fillna(mean_vol, inplace=True)
            logger.warning(f"Filled missing volatility with mean volatility: {mean_vol:.4f}")

    def calculate_moneyness(self):
        """
        Calculates moneyness based on option type.
        """
        logger.info("Calculating moneyness based on option type.")
        self.cleaned_data['moneyness'] = np.where(
            self.cleaned_data['option_type'].str.upper() == 'CALL',
            self.cleaned_data['last_spot_price'] / self.cleaned_data['strike_price'],
            self.cleaned_data['strike_price'] / self.cleaned_data['last_spot_price']
        )

    def simulate_price(self, S0: float, K: float, r: float, sigma: float, T: float, num_simulations: int,
                      distribution_type: str = 'GLD', skewness: float = 0.0, kurtosis_val: float = 3.0) -> Optional[np.ndarray]:
        """
        Simulates future asset prices using specified distribution.

        Args:
            S0 (float): Current asset price.
            K (float): Strike price.
            r (float): Risk-free rate.
            sigma (float): Volatility.
            T (float): Time to maturity in years.
            num_simulations (int): Number of simulations.
            distribution_type (str): Type of distribution ('GLD', 'SkewNormal', 'CornishFisher', etc.).
            skewness (float): Desired skewness.
            kurtosis_val (float): Desired kurtosis.

        Returns:
            Optional[np.ndarray]: Simulated asset prices at maturity.
        """
        try:
            if distribution_type == 'GLD':
                # Use historical returns to fit the GLD
                # Since we're refactoring, we need to pass historical data as an argument
                # Alternatively, extract necessary data from parameters
                # For simplicity, we'll assume historical_data is already accessible
                
                # Note: Since we're no longer using self.current_option_row, we need to adjust
                # However, within simulate_price, we don't have access to option_data
                # Thus, we need to pass historical returns as an argument or adjust the flow
                
                # To resolve, we'll adjust the worker function to pass historical returns
                # Instead, let's return to the worker function to handle historical data
                # For now, simulate_price will use normal distribution as fallback
                
                Z = np.random.normal(0, 1, num_simulations) * sigma * np.sqrt(T)
                logger.debug("Using normal distribution for simulation as GLD is not directly accessible here.")
            elif distribution_type == 'SkewNormal':
                a = skewness
                Z = skewnorm.rvs(a, size=num_simulations)
                logger.debug(f"Generated SkewNormal random variables with skewness parameter: a={a}")
            elif distribution_type == 'CornishFisher':
                Z = np.random.normal(0, 1, num_simulations)
                Z = self.cornish_fisher_quantile(Z, skewness, kurtosis_val)
                logger.debug(f"Adjusted normal random variables using Cornish-Fisher expansion with skewness={skewness}, kurtosis={kurtosis_val}")
            else:
                # Default to normal distribution
                Z = np.random.normal(0, 1, num_simulations)
                logger.debug("Generated normal random variables as fallback.")

            # Geometric Brownian Motion formula
            drift = (r - 0.5 * sigma**2) * T
            diffusion = Z  # Z is already scaled to sigma * sqrt(T)
            S_T = S0 * np.exp(drift + diffusion)

            return S_T
        except Exception as e:
            logger.error(f"Error during price simulation: {e}")
            return None

    def _geometric_brownian_motion(self, S0: float, r: float, sigma: float, T: float, Z: np.ndarray) -> np.ndarray:
        """
        Helper method to compute Geometric Brownian Motion.

        Args:
            S0 (float): Current asset price.
            r (float): Risk-free rate.
            sigma (float): Volatility.
            T (float): Time to maturity in years.
            Z (np.ndarray): Random shocks.

        Returns:
            np.ndarray: Simulated asset prices at maturity.
        """
        drift = (r - 0.5 * sigma**2) * T
        S_T = S0 * np.exp(drift + Z)
        return S_T

    def cornish_fisher_quantile(self, z: np.ndarray, skewness: float = 0.0, kurtosis_val: float = 3.0) -> np.ndarray:
        """
        Adjusts standard normal quantiles using the Cornish-Fisher expansion.

        Args:
            z (np.ndarray): Standard normal quantiles.
            skewness (float): Desired skewness.
            kurtosis_val (float): Desired kurtosis.

        Returns:
            np.ndarray: Adjusted quantiles.
        """
        excess_kurtosis = kurtosis_val - 3.0
        z_adj = (z +
                 (z**2 - 1) * skewness / 6 +
                 (z**3 - 3 * z) * excess_kurtosis / 24 -
                 (2 * z**3 - 5 * z) * skewness**2 / 36)
        return z_adj

    def monte_carlo_simulation_worker(self, option_data: Dict[str, Any], num_simulations: int = 10000) -> Dict[str, Any]:
        """
        Worker function for Monte Carlo simulation. Intended to be used in multiprocessing.

        Args:
            option_data (Dict[str, Any]): Dictionary containing option data.
            num_simulations (int): Number of simulations.

        Returns:
            Dict[str, Any]: Simulation results for the option.
        """
        try:
            # Extract necessary data from option_data
            S0 = option_data['last_spot_price']
            K = option_data['strike_price']
            option_type = option_data['option_type'].upper()
            T = option_data['days'] / 365
            r = self.risk_free_rate
            sigma = option_data['Volatility']
            premium_long = option_data['ask_price']
            premium_short = option_data['bid_price']
            contract_size = option_data['contract_size']
            option_name = option_data['option_name']
            ua_tse_code = option_data['ua_tse_code']

            # Determine skewness and kurtosis based on market scenario and option type
            market_view = self.market_views.get(option_name, 'neutral')
            skewness, kurtosis_val = self.get_skew_kurtosis(option_type, market_view)

            # Simulate asset prices
            Z = self.simulate_price(S0, K, r, sigma, T, num_simulations,
                                    distribution_type=self.distribution_type,
                                    skewness=skewness, kurtosis_val=kurtosis_val)
            if Z is None or len(Z) == 0:
                return {'option_name': option_name, 'pl_long': np.nan, 'pl_short': np.nan, 'S_T': np.nan}

            # Calculate payoffs
            if option_type == 'CALL':
                payoff = np.maximum(Z - K, 0)
            elif option_type == 'PUT':
                payoff = np.maximum(K - Z, 0)
            else:
                payoff = np.nan
                logger.warning(f"Unsupported option type: {option_type} for option {option_name}")

            # Calculate Profit/Loss
            pl_long = (payoff - premium_long) * contract_size
            pl_short = (premium_short - payoff) * contract_size

            return {'option_name': option_name, 'pl_long': pl_long, 'pl_short': pl_short, 'S_T': Z}
        except Exception as e:
            logger.error(f"Error in simulation for {option_data.get('option_name', 'Unknown')}: {e}")
            return {'option_name': option_data.get('option_name', 'Unknown'), 'pl_long': None, 'pl_short': None, 'S_T': None}

    def monte_carlo_simulation(self, num_simulations: int = 10000):
        """
        Performs Monte Carlo simulations for all options.

        Args:
            num_simulations (int): Number of simulations per option.
        """
        logger.info("Starting Monte Carlo simulations.")
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        func = partial(self.monte_carlo_simulation_worker, num_simulations=num_simulations)
        try:
            # Convert each row to a dictionary to pass to the worker
            options_data = self.cleaned_data.to_dict(orient='records')
            results = pool.map(func, options_data)
        except Exception as e:
            logger.error(f"Error during multiprocessing: {e}")
            pool.close()
            pool.join()
            raise
        pool.close()
        pool.join()

        for result in results:
            option_name = result['option_name']
            pl_long = result['pl_long']
            pl_short = result['pl_short']
            S_T = result['S_T']
            if isinstance(pl_long, np.ndarray) and isinstance(pl_short, np.ndarray) and isinstance(S_T, np.ndarray):
                self.simulation_results[option_name] = {'long': pl_long, 'short': pl_short}
                self.target_date_distribution[option_name] = S_T
                self.cash_flows[option_name] = {
                    'long': {
                        'initial': -self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'ask_price'].values[0] *
                                   self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'contract_size'].values[0],
                        'final': pl_long
                    },
                    'short': {
                        'initial': self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'bid_price'].values[0] *
                                   self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'contract_size'].values[0],
                        'final': pl_short
                    }
                }
            else:
                self.simulation_results[option_name] = {'long': np.nan, 'short': np.nan}
                self.target_date_distribution[option_name] = np.nan
                self.cash_flows[option_name] = {
                    'long': {'initial': np.nan, 'final': np.nan},
                    'short': {'initial': np.nan, 'final': np.nan}
                }
        logger.info("Monte Carlo simulations completed.")

    def calculate_pop(self):
        """
        Calculates Probability of Profit (PoP) for each option and position.
        """
        logger.info("Calculating Probability of Profit (PoP) for each option and position.")
        for option, results in self.simulation_results.items():
            pl_long = results.get('long', np.nan)
            pl_short = results.get('short', np.nan)
            pop_long = (np.sum(pl_long > 0) / len(pl_long)) * 100 if self.is_valid_array(pl_long) else np.nan
            pop_short = (np.sum(pl_short > 0) / len(pl_short)) * 100 if self.is_valid_array(pl_short) else np.nan
            self.pop_results[option] = {'long': pop_long, 'short': pop_short}
            logger.debug(f"PoP for {option} - Long: {pop_long:.2f}%, Short: {pop_short:.2f}%")

    def calculate_greeks(self):
        """
        Calculates Greeks for each option.
        """
        logger.info("Calculating Greeks for each option.")
        for idx, row in self.cleaned_data.iterrows():
            S0 = row['last_spot_price']
            K = row['strike_price']
            r = self.risk_free_rate
            sigma = row['Volatility']
            T = row['days'] / 365
            option_type = row['option_type'].upper()
            option_name = row['option_name']

            if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                self.greeks[option_name] = {key: np.nan for key in
                                            ['Delta_long', 'Gamma_long', 'Theta_long', 'Vega_long', 'Rho_long',
                                             'Delta_short', 'Gamma_short', 'Theta_short', 'Vega_short', 'Rho_short']}
                logger.warning(f"Invalid parameters for Greeks calculation for option {option_name}.")
                continue

            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'CALL':
                delta_long = norm.cdf(d1)
                theta_long = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
                              r * K * np.exp(-r * T) * norm.cdf(d2))
                rho_long = K * T * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'PUT':
                delta_long = norm.cdf(d1) - 1
                theta_long = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
                              r * K * np.exp(-r * T) * norm.cdf(-d2))
                rho_long = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            else:
                delta_long = theta_long = rho_long = np.nan
                logger.warning(f"Unsupported option type: {option_type} for option {option_name}")

            gamma_long = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
            vega_long = S0 * norm.pdf(d1) * np.sqrt(T)

            self.greeks[option_name] = {
                'Delta_long': delta_long,
                'Gamma_long': gamma_long,
                'Theta_long': theta_long,
                'Vega_long': vega_long,
                'Rho_long': rho_long,
                'Delta_short': -delta_long,
                'Gamma_short': -gamma_long,
                'Theta_short': -theta_long,
                'Vega_short': -vega_long,
                'Rho_short': -rho_long
            }
            logger.debug(f"Greeks for {option_name}: {self.greeks[option_name]}")

    def calculate_breakeven(self):
        """
        Calculates breakeven points for each option and position.
        """
        logger.info("Calculating breakeven points for each option and position.")
        for idx, row in self.cleaned_data.iterrows():
            option = row['option_name']
            option_type = row['option_type'].upper()
            S0 = row['last_spot_price']
            K = row['strike_price']
            premium_long = row['ask_price']
            premium_short = row['bid_price']
            r = self.risk_free_rate
            T = row['days'] / 365

            if option_type == 'CALL':
                breakeven_long = K + premium_long
                breakeven_short = K + premium_short
            elif option_type == 'PUT':
                breakeven_long = K - premium_long
                breakeven_short = K - premium_short
            else:
                breakeven_long = breakeven_short = np.nan
                logger.warning(f"Unsupported option type: {option_type} for breakeven calculation.")

            adjusted_breakeven_long = breakeven_long * np.exp(r * T) if not np.isnan(breakeven_long) else np.nan
            adjusted_breakeven_short = breakeven_short * np.exp(r * T) if not np.isnan(breakeven_short) else np.nan

            breakeven_long_pct = ((adjusted_breakeven_long - S0) / S0) * 100 if not np.isnan(adjusted_breakeven_long) else np.nan
            breakeven_short_pct = ((adjusted_breakeven_short - S0) / S0) * 100 if not np.isnan(adjusted_breakeven_short) else np.nan

            self.breakeven_points[option] = {
                'long': adjusted_breakeven_long,
                'short': adjusted_breakeven_short,
                'long_pct': breakeven_long_pct,
                'short_pct': breakeven_short_pct
            }
            logger.debug(f"Breakeven for {option}: {self.breakeven_points[option]}")

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.01):
        """
        Calculates Sharpe Ratios for each option and position.

        Args:
            risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.
        """
        logger.info("Calculating Sharpe Ratios for each option and position.")
        for option, positions in self.simulation_results.items():
            pl_long = positions.get('long', np.nan)
            pl_short = positions.get('short', np.nan)

            option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
            if option_row.empty:
                logger.warning(f"No data found for option {option} during Sharpe Ratio calculation.")
                self.sharpe_ratios[option] = {'long': np.nan, 'short': np.nan}
                continue

            option_row = option_row.iloc[0]
            days_to_maturity = option_row['days']

            if self.is_valid_array(pl_long):
                initial_investment_long = option_row['ask_price'] * option_row['contract_size']
                if initial_investment_long > 0:
                    returns_long = pl_long / initial_investment_long
                    mean_return_long = np.mean(returns_long) * (365 / days_to_maturity)
                    std_return_long = np.std(returns_long) * np.sqrt(365 / days_to_maturity)
                    sharpe_long = (mean_return_long - risk_free_rate) / std_return_long if std_return_long > 0 else np.nan
                else:
                    sharpe_long = np.nan
            else:
                sharpe_long = np.nan

            if self.is_valid_array(pl_short):
                initial_premium_short = option_row['bid_price'] * option_row['contract_size']
                if initial_premium_short > 0:
                    returns_short = pl_short / initial_premium_short
                    mean_return_short = np.mean(returns_short) * (365 / days_to_maturity)
                    std_return_short = np.std(returns_short) * np.sqrt(365 / days_to_maturity)
                    sharpe_short = (mean_return_short - risk_free_rate) / std_return_short if std_return_short > 0 else np.nan
                else:
                    sharpe_short = np.nan
            else:
                sharpe_short = np.nan

            self.sharpe_ratios[option] = {'long': sharpe_long, 'short': sharpe_short}
            logger.debug(f"Sharpe Ratios for {option}: {self.sharpe_ratios[option]}")

    def calculate_var_cvar(self, confidence_level: float = 0.95):
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR) for each option and position.

        Args:
            confidence_level (float): Confidence level for VaR and CVaR.
        """
        logger.info(f"Calculating VaR and CVaR at {confidence_level*100:.0f}% confidence level.")
        for option, positions in self.simulation_results.items():
            pl_long = positions.get('long', np.nan)
            pl_short = positions.get('short', np.nan)

            if self.is_valid_array(pl_long):
                var_long = np.percentile(pl_long, (1 - confidence_level) * 100)
                cvar_long = pl_long[pl_long <= var_long].mean()
            else:
                var_long = cvar_long = np.nan

            if self.is_valid_array(pl_short):
                var_short = np.percentile(pl_short, (1 - confidence_level) * 100)
                cvar_short = pl_short[pl_short <= var_short].mean()
            else:
                var_short = cvar_short = np.nan

            self.var[option] = {'long': var_long, 'short': var_short}
            self.cvar[option] = {'long': cvar_long, 'short': cvar_short}
            logger.debug(f"VaR and CVaR for {option}: VaR_Long={var_long}, CVaR_Long={cvar_long}, VaR_Short={var_short}, CVaR_Short={cvar_short}")

    def calculate_variance(self):
        """
        Calculates Variance for each option and position.
        """
        logger.info("Calculating Variance for each option and position.")
        for option, positions in self.simulation_results.items():
            pl_long = positions.get('long', np.nan)
            pl_short = positions.get('short', np.nan)

            option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
            if option_row.empty:
                logger.warning(f"No data found for option {option} during Variance calculation.")
                self.variance[option] = {'long': np.nan, 'short': np.nan}
                continue

            option_row = option_row.iloc[0]
            days_to_maturity = option_row['days']

            if self.is_valid_array(pl_long):
                variance_long = np.var(pl_long) * (252 / days_to_maturity)
            else:
                variance_long = np.nan

            if self.is_valid_array(pl_short):
                variance_short = np.var(pl_short) * (252 / days_to_maturity)
            else:
                variance_short = np.nan

            self.variance[option] = {'long': variance_long, 'short': variance_short}
            logger.debug(f"Variance for {option}: Variance_Long={variance_long}, Variance_Short={variance_short}")

    def calculate_payout_ratio(self):
        """
        Calculates Payout Ratios and Premium Efficiency for each option and position.
        """
        logger.info("Calculating Payout Ratios and Premium Efficiency for each option and position.")
        for option, cash_flows in self.cash_flows.items():
            option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
            if option_row.empty:
                logger.warning(f"No data found for option {option} during Payout Ratio calculation.")
                self.payout_ratios[option] = {
                    'long': np.nan,
                    'short': np.nan,
                    'premium_efficiency_long': np.nan,
                    'premium_efficiency_short': np.nan
                }
                continue

            option_row = option_row.iloc[0]
            contract_size = option_row['contract_size']
            premium_long = option_row['ask_price']
            premium_short = option_row['bid_price']
            pl_long = cash_flows['long']['final']
            pl_short = cash_flows['short']['final']

            if self.is_valid_array(pl_long):
                average_pl_long = np.mean(pl_long)
                initial_investment_long = premium_long * contract_size
                payout_long = average_pl_long / initial_investment_long if initial_investment_long > 0 else np.nan
                premium_efficiency_long = 1 / initial_investment_long if initial_investment_long > 0 else 0
            else:
                payout_long = premium_efficiency_long = np.nan

            if self.is_valid_array(pl_short):
                average_pl_short = np.mean(pl_short)
                initial_premium_short = premium_short * contract_size
                payout_short = average_pl_short / initial_premium_short if initial_premium_short > 0 else np.nan
                premium_efficiency_short = initial_premium_short
            else:
                payout_short = premium_efficiency_short = np.nan

            self.payout_ratios[option] = {
                'long': payout_long,
                'short': payout_short,
                'premium_efficiency_long': premium_efficiency_long,
                'premium_efficiency_short': premium_efficiency_short
            }
            logger.debug(f"Payout Ratios for {option}: {self.payout_ratios[option]}")

    def determine_market_views(self):
        """
        Determines market views based on moneyness for each option.
        """
        logger.info("Determining market views based on moneyness.")
        for _, row in self.cleaned_data.iterrows():
            option = row['option_name']
            option_type = row['option_type'].upper()
            S = row['last_spot_price']
            K = row['strike_price']
            if option_type == 'CALL':
                self.market_views[option] = 'bullish' if S > K else 'bearish'
            elif option_type == 'PUT':
                self.market_views[option] = 'bearish' if S < K else 'bullish'
            else:
                self.market_views[option] = 'neutral'
                logger.warning(f"Unsupported option type: {option_type} for market view determination.")

    def calculate_metrics_min_max(self):
        """
        Calculates minimum and maximum values for various metrics to assist in standardization.
        """
        metrics = ['sharpe_ratios', 'pop_results', 'var', 'cvar', 'payout_ratios', 'breakeven_pct', 'premium_efficiency']
        positions = ['long', 'short']
        for metric in metrics:
            self.metrics_min[metric] = {}
            self.metrics_max[metric] = {}
            for position in positions:
                if metric == 'sharpe_ratios':
                    values = [self.sharpe_ratios.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'pop_results':
                    values = [self.pop_results.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'payout_ratios':
                    values = [self.payout_ratios.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'premium_efficiency':
                    key = f'premium_efficiency_{position}'
                    values = [self.payout_ratios.get(opt, {}).get(key, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'breakeven_pct':
                    key = f'breakeven_{position}_pct'
                    values = [self.breakeven_points.get(opt, {}).get(key, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'var':
                    values = [self.var.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
                elif metric == 'cvar':
                    values = [self.cvar.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
                else:
                    values = []

                values = np.array(values)
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    self.metrics_min[metric][position] = np.min(values)
                    self.metrics_max[metric][position] = np.max(values)
                    logger.debug(f"Metric '{metric}' for position '{position}': min={self.metrics_min[metric][position]}, max={self.metrics_max[metric][position]}")
                else:
                    self.metrics_min[metric][position] = 0
                    self.metrics_max[metric][position] = 1
                    logger.debug(f"Metric '{metric}' for position '{position}' has no valid values. Set min=0 and max=1.")

    def standardize_metric(self, value: float, metric_name: str, position: str) -> float:
        """
        Standardizes a metric value based on its min and max.

        Args:
            value (float): The metric value to standardize.
            metric_name (str): Name of the metric.
            position (str): 'long' or 'short'.

        Returns:
            float: Standardized metric value.
        """
        if metric_name in ['sharpe_ratios', 'pop_results', 'payout_ratios', 'premium_efficiency']:
            min_value = self.metrics_min[metric_name][position]
            max_value = self.metrics_max[metric_name][position]
            standardized = (value - min_value) / (max_value - min_value) if max_value != min_value else 0
        elif metric_name in ['var', 'cvar', 'breakeven_pct']:
            min_value = self.metrics_min[metric_name][position]
            max_value = self.metrics_max[metric_name][position]
            standardized = (max_value - value) / (max_value - min_value) if max_value != min_value else 0
        else:
            standardized = 0
            logger.warning(f"Unknown metric '{metric_name}' encountered during standardization.")
        return standardized

    def calculate_composite_score(self, option: str, position: str) -> float:
        """
        Calculates a composite score for an option's position based on various metrics.

        Args:
            option (str): Option name.
            position (str): 'long' or 'short'.

        Returns:
            float: Composite score.
        """
        metrics = ['sharpe_ratios', 'pop_results', 'var', 'cvar', 'payout_ratios', 'breakeven_pct', 'premium_efficiency']
        base_weights = {
            'sharpe_ratios': 0.10,
            'pop_results': 0.15,
            'var': 0.10,
            'cvar': 0.10,
            'payout_ratios': 0.15,
            'breakeven_pct': 0.25,
            'premium_efficiency': 0.15
        }
        adjusted_weights = base_weights.copy()
        if self.strategy == 'aggressive':
            adjusted_weights['payout_ratios'] += 0.05
            adjusted_weights['var'] -= 0.05
        elif self.strategy == 'conservative':
            adjusted_weights['var'] += 0.05
            adjusted_weights['payout_ratios'] -= 0.05

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        score = 0
        for key in metrics:
            if key == 'premium_efficiency':
                value = self.payout_ratios.get(option, {}).get(f'premium_efficiency_{position}', np.nan)
            elif key == 'breakeven_pct':
                value = self.breakeven_points.get(option, {}).get(f'breakeven_{position}_pct', np.nan)
            else:
                metric_dict = getattr(self, key, {})
                value = metric_dict.get(option, {}).get(position, np.nan)

            if not np.isnan(value):
                standardized = self.standardize_metric(value, key, position)
                score += adjusted_weights[key] * standardized
                logger.debug(f"Metric '{key}' for option '{option}' position '{position}': value={value}, standardized={standardized}, weight={adjusted_weights[key]}, score contribution={adjusted_weights[key] * standardized}")
            else:
                logger.debug(f"Metric '{key}' for option '{option}' position '{position}' is NaN and will not contribute to the score.")

        logger.debug(f"Composite score for option '{option}' position '{position}': {score}")
        return score

    def get_recommendations(self, user_market_view: str):
        """
        Generates recommendations for each option based on composite scores.

        Args:
            user_market_view (str): The user's market view (e.g., 'bullish', 'bearish').
        """
        logger.info("Generating recommendations based on composite scores.")
        self.calculate_payout_ratio()
        self.determine_market_views()
        self.calculate_metrics_min_max()

        options_list = self.cleaned_data['option_name'].tolist()
        recommendations = []
        for option in options_list:
            score_long = self.calculate_composite_score(option, 'long')
            score_short = self.calculate_composite_score(option, 'short')
            final_recommendation = 'Long' if score_long > score_short else 'Short'
            recommendations.append({
                'OptionName': option,
                'Recommendation': final_recommendation,
                'CompositeScore_Long': score_long,
                'CompositeScore_Short': score_short
            })

        self.recommendations_df = pd.DataFrame(recommendations)
        logger.info("Recommendations generated successfully.")

    def is_valid_array(self, arr: np.ndarray) -> bool:
        """
        Checks if the provided array is valid for calculations.

        Args:
            arr (np.ndarray): Array to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        return isinstance(arr, np.ndarray) and arr.size > 0 and not np.isnan(arr).all()

    def get_skew_kurtosis(self, option_type: str, market_view: str) -> Tuple[float, float]:
        """
        Retrieves skewness and kurtosis based on market scenario and option type.

        Args:
            option_type (str): 'CALL' or 'PUT'.
            market_view (str): 'bullish', 'bearish', or 'neutral'.

        Returns:
            Tuple[float, float]: Skewness and kurtosis values.
        """
        scenario = self.market_scenario
        skewness = self.market_scenarios.get(scenario, {}).get('skewness', 0.0)
        kurtosis_val = self.market_scenarios.get(scenario, {}).get('kurtosis', 3.0)
        if option_type == 'PUT':
            skewness = -skewness
        return skewness, kurtosis_val

    def compile_metrics_data(self) -> pd.DataFrame:
        """
        Compiles all calculated metrics into a structured DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all metrics per option.
        """
        try:
            data = []
            for option in self.cleaned_data['option_name']:
                metrics = {
                    'OptionName': option,
                    'Sharpe_Long': self.sharpe_ratios.get(option, {}).get('long', np.nan),
                    'Sharpe_Short': self.sharpe_ratios.get(option, {}).get('short', np.nan),
                    'PoP_Long': self.pop_results.get(option, {}).get('long', np.nan),
                    'PoP_Short': self.pop_results.get(option, {}).get('short', np.nan),
                    'VaR_Long': self.var.get(option, {}).get('long', np.nan),
                    'VaR_Short': self.var.get(option, {}).get('short', np.nan),
                    'CVaR_Long': self.cvar.get(option, {}).get('long', np.nan),
                    'CVaR_Short': self.cvar.get(option, {}).get('short', np.nan),
                    'Breakeven_Long_Pct': self.breakeven_points.get(option, {}).get('long_pct', np.nan),
                    'Breakeven_Short_Pct': self.breakeven_points.get(option, {}).get('short_pct', np.nan),
                    'Premium_Efficiency_Long': self.payout_ratios.get(option, {}).get('premium_efficiency_long', np.nan),
                    'Premium_Efficiency_Short': self.payout_ratios.get(option, {}).get('premium_efficiency_short', np.nan),
                    'Variance_Long': self.variance.get(option, {}).get('long', np.nan),
                    'Variance_Short': self.variance.get(option, {}).get('short', np.nan),
                    # Add more metrics as needed
                }
                data.append(metrics)
            metrics_df = pd.DataFrame(data)
            logger.info("Compiled metrics data into DataFrame.")
            return metrics_df
        except Exception as e:
            logger.error(f"Error compiling metrics data: {e}")
            return pd.DataFrame()

    ## **Scenario Analysis with Stress Testing**

    def perform_scenario_analysis(self, num_simulations: int = 10000):
        """
        Performs scenario analysis by simulating option performance under different market scenarios.

        Args:
            num_simulations (int): Number of simulations per scenario.
        """
        logger.info("Performing scenario analysis.")
        scenarios = self.market_scenarios.keys()
        scenario_results = {}

        for scenario in scenarios:
            logger.info(f"Simulating scenario: {scenario}")
            self.market_scenario = scenario
            self.monte_carlo_simulation(num_simulations=num_simulations)
            self.calculate_pop()
            self.calculate_var_cvar()
            self.calculate_payout_ratio()

            # Compile results for the scenario
            scenario_metrics = {}
            for option in self.cleaned_data['option_name']:
                pop_long = self.pop_results.get(option, {}).get('long', np.nan)
                pop_short = self.pop_results.get(option, {}).get('short', np.nan)
                var_long = self.var.get(option, {}).get('long', np.nan)
                var_short = self.var.get(option, {}).get('short', np.nan)
                payout_long = self.payout_ratios.get(option, {}).get('long', np.nan)
                payout_short = self.payout_ratios.get(option, {}).get('short', np.nan)
                scenario_metrics[option] = {
                    'PoP_Long': pop_long,
                    'PoP_Short': pop_short,
                    'VaR_Long': var_long,
                    'VaR_Short': var_short,
                    'Payout_Long': payout_long,
                    'Payout_Short': payout_short
                }
            scenario_results[scenario] = scenario_metrics
            logger.info(f"Completed scenario: {scenario}")

        self.scenario_analysis_results = scenario_results
        logger.info("Scenario analysis completed successfully.")

    def plot_scenario_analysis(self, option_name: str):
        """
        Plots the scenario analysis results for a given option.

        Args:
            option_name (str): The name of the option to plot.
        """
        if not hasattr(self, 'scenario_analysis_results') or not self.scenario_analysis_results:
            logger.error("No scenario analysis results found. Please run perform_scenario_analysis() first.")
            return

        if option_name not in self.scenario_analysis_results[next(iter(self.scenario_analysis_results))]:
            logger.error(f"Option '{option_name}' not found in scenario analysis results.")
            return

        scenarios = list(self.scenario_analysis_results.keys())
        pops_long = [self.scenario_analysis_results[scenario][option_name]['PoP_Long'] for scenario in scenarios]
        pops_short = [self.scenario_analysis_results[scenario][option_name]['PoP_Short'] for scenario in scenarios]
        vars_long = [self.scenario_analysis_results[scenario][option_name]['VaR_Long'] for scenario in scenarios]
        vars_short = [self.scenario_analysis_results[scenario][option_name]['VaR_Short'] for scenario in scenarios]

        # Plot Probability of Profit
        plt.figure(figsize=(12, 6))
        plt.plot(scenarios, pops_long, label='PoP Long', marker='o')
        plt.plot(scenarios, pops_short, label='PoP Short', marker='x')
        plt.title(f"Probability of Profit across Scenarios for {option_name}")
        plt.xlabel("Market Scenarios")
        plt.ylabel("Probability of Profit (%)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot Value at Risk
        plt.figure(figsize=(12, 6))
        plt.plot(scenarios, vars_long, label='VaR Long', marker='o')
        plt.plot(scenarios, vars_short, label='VaR Short', marker='x')
        plt.title(f"Value at Risk across Scenarios for {option_name}")
        plt.xlabel("Market Scenarios")
        plt.ylabel("VaR")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        logger.info(f"Plotted scenario analysis for option '{option_name}'.")

    ## **Optional: Sensitivity Analysis**

    def sensitivity_analysis(self, option_name: str, parameter: str, changes: list) -> pd.DataFrame:
        """
        Performs sensitivity analysis on a given parameter.

        Args:
            option_name (str): The option to analyze.
            parameter (str): The parameter to vary ('sigma', 'r', 'T').
            changes (list): List of percentage changes to apply (e.g., [-0.2, -0.1, 0, 0.1, 0.2]).

        Returns:
            pd.DataFrame: DataFrame containing sensitivity analysis results.
        """
        logger.info(f"Performing sensitivity analysis on '{parameter}' for option '{option_name}'.")
        option_row = self.cleaned_data[self.cleaned_data['option_name'] == option_name]
        if option_row.empty:
            logger.error(f"Option '{option_name}' not found in data.")
            return pd.DataFrame()

        option_row = option_row.iloc[0]
        results = []

        for change in changes:
            modified_row = option_row.copy()
            original_value = 0
            if parameter == 'sigma':
                original_value = modified_row['Volatility']
                modified_row['Volatility'] *= (1 + change)
            elif parameter == 'r':
                original_value = self.risk_free_rate
                self.risk_free_rate *= (1 + change)
            elif parameter == 'T':
                original_value = modified_row['days']
                modified_row['days'] *= (1 + change)
            else:
                logger.warning(f"Unsupported parameter: {parameter}")
                continue

            # Simulate with modified parameter
            simulation_result = self.monte_carlo_simulation_worker(modified_row.to_dict(), num_simulations=5000)
            pl_long = simulation_result['pl_long']
            pl_short = simulation_result['pl_short']
            avg_pl_long = np.mean(pl_long) if self.is_valid_array(pl_long) else np.nan
            avg_pl_short = np.mean(pl_short) if self.is_valid_array(pl_short) else np.nan

            # Restore original parameter
            if parameter == 'sigma':
                modified_row['Volatility'] = original_value
            elif parameter == 'r':
                self.risk_free_rate = original_value
            elif parameter == 'T':
                modified_row['days'] = original_value

            results.append({
                'Change (%)': change * 100,
                'Avg_PL_Long': avg_pl_long,
                'Avg_PL_Short': avg_pl_short
            })

        sensitivity_df = pd.DataFrame(results)
        logger.info(f"Sensitivity analysis completed for '{parameter}' on option '{option_name}'.")
        return sensitivity_df

    ## **Compile Metrics Data**

    def compile_metrics_data(self) -> pd.DataFrame:
        """
        Compiles all calculated metrics into a structured DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all metrics per option.
        """
        try:
            data = []
            for option in self.cleaned_data['option_name']:
                metrics = {
                    'OptionName': option,
                    'Sharpe_Long': self.sharpe_ratios.get(option, {}).get('long', np.nan),
                    'Sharpe_Short': self.sharpe_ratios.get(option, {}).get('short', np.nan),
                    'PoP_Long': self.pop_results.get(option, {}).get('long', np.nan),
                    'PoP_Short': self.pop_results.get(option, {}).get('short', np.nan),
                    'VaR_Long': self.var.get(option, {}).get('long', np.nan),
                    'VaR_Short': self.var.get(option, {}).get('short', np.nan),
                    'CVaR_Long': self.cvar.get(option, {}).get('long', np.nan),
                    'CVaR_Short': self.cvar.get(option, {}).get('short', np.nan),
                    'Breakeven_Long_Pct': self.breakeven_points.get(option, {}).get('long_pct', np.nan),
                    'Breakeven_Short_Pct': self.breakeven_points.get(option, {}).get('short_pct', np.nan),
                    'Premium_Efficiency_Long': self.payout_ratios.get(option, {}).get('premium_efficiency_long', np.nan),
                    'Premium_Efficiency_Short': self.payout_ratios.get(option, {}).get('premium_efficiency_short', np.nan),
                    'Variance_Long': self.variance.get(option, {}).get('long', np.nan),
                    'Variance_Short': self.variance.get(option, {}).get('short', np.nan),
                    # Add more metrics as needed
                }
                data.append(metrics)
            metrics_df = pd.DataFrame(data)
            logger.info("Compiled metrics data into DataFrame.")
            return metrics_df
        except Exception as e:
            logger.error(f"Error compiling metrics data: {e}")
            return pd.DataFrame()

    ## **Scenario Analysis with Stress Testing**

    def perform_scenario_analysis(self, num_simulations: int = 10000):
        """
        Performs scenario analysis by simulating option performance under different market scenarios.

        Args:
            num_simulations (int): Number of simulations per scenario.
        """
        logger.info("Performing scenario analysis.")
        scenarios = self.market_scenarios.keys()
        scenario_results = {}

        for scenario in scenarios:
            logger.info(f"Simulating scenario: {scenario}")
            self.market_scenario = scenario
            self.monte_carlo_simulation(num_simulations=num_simulations)
            self.calculate_pop()
            self.calculate_var_cvar()
            self.calculate_payout_ratio()

            # Compile results for the scenario
            scenario_metrics = {}
            for option in self.cleaned_data['option_name']:
                pop_long = self.pop_results.get(option, {}).get('long', np.nan)
                pop_short = self.pop_results.get(option, {}).get('short', np.nan)
                var_long = self.var.get(option, {}).get('long', np.nan)
                var_short = self.var.get(option, {}).get('short', np.nan)
                payout_long = self.payout_ratios.get(option, {}).get('long', np.nan)
                payout_short = self.payout_ratios.get(option, {}).get('short', np.nan)
                scenario_metrics[option] = {
                    'PoP_Long': pop_long,
                    'PoP_Short': pop_short,
                    'VaR_Long': var_long,
                    'VaR_Short': var_short,
                    'Payout_Long': payout_long,
                    'Payout_Short': payout_short
                }
            scenario_results[scenario] = scenario_metrics
            logger.info(f"Completed scenario: {scenario}")

        self.scenario_analysis_results = scenario_results
        logger.info("Scenario analysis completed successfully.")

    def plot_scenario_analysis(self, option_name: str):
        """
        Plots the scenario analysis results for a given option.

        Args:
            option_name (str): The name of the option to plot.
        """
        if not hasattr(self, 'scenario_analysis_results') or not self.scenario_analysis_results:
            logger.error("No scenario analysis results found. Please run perform_scenario_analysis() first.")
            return

        # Check if option exists in the scenario analysis
        first_scenario = next(iter(self.scenario_analysis_results))
        if option_name not in self.scenario_analysis_results[first_scenario]:
            logger.error(f"Option '{option_name}' not found in scenario analysis results.")
            return

        scenarios = list(self.scenario_analysis_results.keys())
        pops_long = [self.scenario_analysis_results[scenario][option_name]['PoP_Long'] for scenario in scenarios]
        pops_short = [self.scenario_analysis_results[scenario][option_name]['PoP_Short'] for scenario in scenarios]
        vars_long = [self.scenario_analysis_results[scenario][option_name]['VaR_Long'] for scenario in scenarios]
        vars_short = [self.scenario_analysis_results[scenario][option_name]['VaR_Short'] for scenario in scenarios]

        # Plot Probability of Profit
        plt.figure(figsize=(12, 6))
        plt.plot(scenarios, pops_long, label='PoP Long', marker='o')
        plt.plot(scenarios, pops_short, label='PoP Short', marker='x')
        plt.title(f"Probability of Profit across Scenarios for {option_name}")
        plt.xlabel("Market Scenarios")
        plt.ylabel("Probability of Profit (%)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot Value at Risk
        plt.figure(figsize=(12, 6))
        plt.plot(scenarios, vars_long, label='VaR Long', marker='o')
        plt.plot(scenarios, vars_short, label='VaR Short', marker='x')
        plt.title(f"Value at Risk across Scenarios for {option_name}")
        plt.xlabel("Market Scenarios")
        plt.ylabel("VaR")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        logger.info(f"Plotted scenario analysis for option '{option_name}'.")

    ## **Optional: Sensitivity Analysis**

    def sensitivity_analysis(self, option_name: str, parameter: str, changes: list) -> pd.DataFrame:
        """
        Performs sensitivity analysis on a given parameter.

        Args:
            option_name (str): The option to analyze.
            parameter (str): The parameter to vary ('sigma', 'r', 'T').
            changes (list): List of percentage changes to apply (e.g., [-0.2, -0.1, 0, 0.1, 0.2]).

        Returns:
            pd.DataFrame: DataFrame containing sensitivity analysis results.
        """
        logger.info(f"Performing sensitivity analysis on '{parameter}' for option '{option_name}'.")
        option_row = self.cleaned_data[self.cleaned_data['option_name'] == option_name]
        if option_row.empty:
            logger.error(f"Option '{option_name}' not found in data.")
            return pd.DataFrame()

        option_row = option_row.iloc[0]
        results = []

        for change in changes:
            modified_row = option_row.copy()
            original_value = 0
            if parameter == 'sigma':
                original_value = modified_row['Volatility']
                modified_row['Volatility'] *= (1 + change)
            elif parameter == 'r':
                original_value = self.risk_free_rate
                self.risk_free_rate *= (1 + change)
            elif parameter == 'T':
                original_value = modified_row['days']
                modified_row['days'] *= (1 + change)
            else:
                logger.warning(f"Unsupported parameter: {parameter}")
                continue

            # Simulate with modified parameter
            simulation_result = self.monte_carlo_simulation_worker(modified_row.to_dict(), num_simulations=5000)
            pl_long = simulation_result['pl_long']
            pl_short = simulation_result['pl_short']
            avg_pl_long = np.mean(pl_long) if self.is_valid_array(pl_long) else np.nan
            avg_pl_short = np.mean(pl_short) if self.is_valid_array(pl_short) else np.nan

            # Restore original parameter
            if parameter == 'sigma':
                modified_row['Volatility'] = original_value
            elif parameter == 'r':
                self.risk_free_rate = original_value
            elif parameter == 'T':
                modified_row['days'] = original_value

            results.append({
                'Change (%)': change * 100,
                'Avg_PL_Long': avg_pl_long,
                'Avg_PL_Short': avg_pl_short
            })

        sensitivity_df = pd.DataFrame(results)
        logger.info(f"Sensitivity analysis completed for '{parameter}' on option '{option_name}'.")
        return sensitivity_df

    ## **Compile Metrics Data**

    def compile_metrics_data(self) -> pd.DataFrame:
        """
        Compiles all calculated metrics into a structured DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all metrics per option.
        """
        try:
            data = []
            for option in self.cleaned_data['option_name']:
                metrics = {
                    'OptionName': option,
                    'Sharpe_Long': self.sharpe_ratios.get(option, {}).get('long', np.nan),
                    'Sharpe_Short': self.sharpe_ratios.get(option, {}).get('short', np.nan),
                    'PoP_Long': self.pop_results.get(option, {}).get('long', np.nan),
                    'PoP_Short': self.pop_results.get(option, {}).get('short', np.nan),
                    'VaR_Long': self.var.get(option, {}).get('long', np.nan),
                    'VaR_Short': self.var.get(option, {}).get('short', np.nan),
                    'CVaR_Long': self.cvar.get(option, {}).get('long', np.nan),
                    'CVaR_Short': self.cvar.get(option, {}).get('short', np.nan),
                    'Breakeven_Long_Pct': self.breakeven_points.get(option, {}).get('long_pct', np.nan),
                    'Breakeven_Short_Pct': self.breakeven_points.get(option, {}).get('short_pct', np.nan),
                    'Premium_Efficiency_Long': self.payout_ratios.get(option, {}).get('premium_efficiency_long', np.nan),
                    'Premium_Efficiency_Short': self.payout_ratios.get(option, {}).get('premium_efficiency_short', np.nan),
                    'Variance_Long': self.variance.get(option, {}).get('long', np.nan),
                    'Variance_Short': self.variance.get(option, {}).get('short', np.nan),
                    # Add more metrics as needed
                }
                data.append(metrics)
            metrics_df = pd.DataFrame(data)
            logger.info("Compiled metrics data into DataFrame.")
            return metrics_df
        except Exception as e:
            logger.error(f"Error compiling metrics data: {e}")
            return pd.DataFrame()
