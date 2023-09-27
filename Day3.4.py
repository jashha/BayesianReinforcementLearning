import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Normal
import logging
import random
from collections import deque
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from torch.nn import functional as F
import torch
import torch.nn as nn
from scipy.stats import beta
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch
import torch.nn as nn
import torch.distributions as dist
from pyro.infer import TraceEnum_ELBO, config_enumerate
import pyro




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)

'''
Define env
'''





class BayesianLayer111(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn='leakyselu'):
        super(BayesianLayer111, self).__init__()
        
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))
        
        if activation_fn == 'leakyselu':
            self.activation = nn.LeakyReLU()
        elif activation_fn == 'selu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        weight_sample = Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias_sample = Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()
        output = F.linear(x, weight_sample, bias_sample)
        output = self.activation(output)
        return output

class BayesianLayer222(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLayer222, self).__init__()
        
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))

    def forward(self, x):
        weight_sample = Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias_sample = Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()
        output = F.linear(x, weight_sample, bias_sample)
        return output


class BayesianRewardCalculator:
    def __init__(self, initial_balance, transaction_fee):
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.returns = []
        # Initialize other parameters for Bayesian calculation

    def update(self, new_return):
        self.returns.append(new_return)
        # Perform Bayesian calculations

    def get_reward(self):
        # Calculate and return the Bayesian Sharpe Ratio
        pass



class BayesianPortfolioManager:
    def __init__(self, initial_balance, transaction_fee):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_fee = transaction_fee
        self.returns = []  # Initializing as a list

    def reset(self):
        self.balance = self.initial_balance
        self.returns = []  # Resetting to an empty list
        return self.get_state()

    def calculate_reward(self):
        daily_return = (self.balance / self.initial_balance) - 1
        sharpe_ratio = self.calculate_bayesian_sharpe_ratio(daily_return)
        return sharpe_ratio

    def execute_trade(self, asset_price, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        
        amount_to_buy_or_sell = asset_price * action - self.transaction_fee
        self.balance += amount_to_buy_or_sell  # Updating the balance based on the trade

    def calculate_bayesian_sharpe_ratio(self, daily_return):
        # Your Bayesian Sharpe Ratio calculation here
        return daily_return  # Placeholder. Replace with actual Bayesian Sharpe Ratio.

    def update_returns(self, daily_return):
        self.returns.append(daily_return)
##########################################################

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class BayesianNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SharpeRatioBayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Define alpha and beta as learnable parameters
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
    
    def model(self, x, y):
        # Use learned alpha and beta parameters
        alpha = self.alpha
        beta = self.beta

        # Rest of your model remains unchanged
        lhat = torch.sigmoid(lifted_reg_model(x) * alpha + beta)
        
        fc1w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
        fc1b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
        
        fc2w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
        fc2b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))

        fc3w_prior = dist.Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight))
        fc3b_prior = dist.Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias))
        
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()

        lhat = torch.sigmoid(lifted_reg_model(x))

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Bernoulli(lhat), obs=y)
            
    def guide(self, x, y):

        # Define priors for alpha and beta as normal distributions
        alpha_mu = pyro.param("alpha_mu", torch.randn(1))
        alpha_sigma = pyro.param("alpha_sigma", torch.randn(1).exp())
        beta_mu = pyro.param("beta_mu", torch.randn(1))
        beta_sigma = pyro.param("beta_sigma", torch.randn(1).exp())

        # Sample alpha and beta from the defined priors
        alpha = pyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
        beta = pyro.sample("beta", dist.Normal(beta_mu, beta_sigma))


        
        fc1w_mu = torch.randn_like(self.fc1.weight)
        fc1w_sigma = torch.randn_like(self.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = pyro.param("fc1w_sigma", fc1w_sigma)
        fc1w_prior = dist.Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        
        fc1b_mu = torch.randn_like(self.fc1.bias)
        fc1b_sigma = torch.randn_like(self.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = pyro.param("fc1b_sigma", fc1b_sigma)
        fc1b_prior = dist.Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # For fc2
        fc2w_mu = torch.randn_like(self.fc2.weight)
        fc2w_sigma = torch.randn_like(self.fc2.weight)
        fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
        fc2w_sigma_param = pyro.param("fc2w_sigma", fc2w_sigma)
        fc2w_prior = dist.Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
        
        fc2b_mu = torch.randn_like(self.fc2.bias)
        fc2b_sigma = torch.randn_like(self.fc2.bias)
        fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
        fc2b_sigma_param = pyro.param("fc2b_sigma", fc2b_sigma)
        fc2b_prior = dist.Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
        
        # And fc3 layers
        fc3w_mu = torch.randn_like(self.fc3.weight)
        fc3w_sigma = torch.randn_like(self.fc3.weight)
        fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
        fc3w_sigma_param = pyro.param("fc3w_sigma", fc3w_sigma)
        fc3w_prior = dist.Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)

        
        fc3b_mu = torch.randn_like(self.fc3.bias)
        fc3b_sigma = torch.randn_like(self.fc3.bias)
        fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
        fc3b_sigma_param = pyro.param("fc3b_sigma", fc3b_sigma)
        fc3b_prior = dist.Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)
        
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, # Add fc2...
                  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, # And fc3...
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior
                  }
        
        lifted_module = pyro.random_module("module", self, priors)
        
        return lifted_module()

    def infer(self, x, y, num_iterations=1000):
        pyro.clear_param_store()
        optim = Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        for j in range(num_iterations):
            loss = svi.step(x, y)
            if j % 100 == 0:
                print(f"[iteration {j+1}] loss: {loss}")

        # After SVI, you can retrieve the tuned values of alpha and beta
        tuned_alpha = self.alpha.item()
        tuned_beta = self.beta.item()
        
        return tuned_alpha, tuned_beta

##################################################


# Add these imports at the beginning of your code
class BayesianLayer22(BayesianNN2):
    #outputs a layer
    def __init__(self, input_dim, output_dim, activation_fn='leakyselu'):
        super(BayesianLayer22, self).__init__()
        
        # Initialize means and standard deviations for weights
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        
        # Initialize means and standard deviations for biases
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))
        
        # Activation function
        if activation_fn == 'leakyselu':
            self.activation = nn.LeakyReLU()
        elif activation_fn == 'selu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        # Add more activation functions as needed
        
    def forward(self, x):
        # Sample weights and biases from their respective posterior distributions
        weight_sample = Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias_sample = Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()
        
        # Compute the forward pass using the sampled weights and biases
        output = F.linear(x, weight_sample, bias_sample)
        
        # Apply the activation function
        output = self.activation(output)

        return output

class BayesianLayer11(BayesianNN2):
    # outputs mean std
    def __init__(self, input_dim, output_dim):
        super(BayesianLayer11, self).__init__()
        
        # Initialize means and standard deviations for weights
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        
        # Initialize means and standard deviations for biases
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))

    def forward(self, x):
        # Sample weights and biases from their respective posterior distributions
        weight_sample = Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias_sample = Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()

        # Compute the forward pass using the sampled weights and biases
        output = F.linear(x, weight_sample, bias_sample)

        return output

##########################################
    
'''
Define env
'''
##class CryptoTradingEnv:
##    def __init__(self, data, initial_balance=1000, transaction_fee=0.001, start_step=4000):
##        self.data = data
##        self.start_step = start_step
##        self.current_step = start_step
##        self.initial_balance = initial_balance
##        self.transaction_fee = transaction_fee
##        self.balance = initial_balance
##        self.returns = []  # Initializing as a list
##
##    def reset(self):
##        self.balance = self.initial_balance
##        self.current_step = 4000
##        self.returns = []  # Resetting to an empty list
##        self.initial_balance = 10000
##        self.current_step = 4000
##        return self.get_state()
##
##
##
####class CryptoTradingEnv:
####    def __init__(self, data, initial_balance=10000, transaction_fee=0.001, start_step=4000):
####        self.data = data
####        self.start_step = start_step
####        self.current_step = start_step
####        self.portfolio_manager = BayesianPortfolioManager(initial_balance, transaction_fee)
####
####
####        
####    def reset(self):
####        self.current_step = 4000
####        return self.portfolio_manager.reset()
##
##    def get_state(self):
##        row = self.data.iloc[self.current_step]
##        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
##        return np.array([float(self.balance)] + float_values)
##
##    def step(self, actions):
##        self.current_step += 1
##        if self.current_step >= len(self.data):
##            return self.reset(), 0, True
##
##        for idx, action in enumerate(actions):
##            self.execute_trade(idx, action)
##
##        # Update BayesianRewardCalculator
##        '''
##        new_return = (self.balance / self.initial_balance) - 1
##        self.bayesian_reward_calculator.update(new_return)
##
##        reward = self.bayesian_reward_calculator.get_reward()
##    
##        for idx, action in enumerate(actions):
##            self.execute_trade(idx, action)
##
##
##        '''
##        reward = self.calculate_reward()
##        self.returns.append(reward)
##        done = self.current_step >= len(self.data) - 1
##
##        return self.get_state(), reward, done
##
##
##    def execute_trade(self, asset_idx, action):
##        if isinstance(action, torch.Tensor):
##            action = action.detach().numpy()
##
##        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'  # Or use logic to find it dynamically
##
##        try:
##            price = self.data.iloc[self.current_step][asset_name]
##        except KeyError:
##            print(f"Column {asset_name} not found in DataFrame.")
##            return
##
##        logging.info(f'price = {price} | action = {action}')
##        amount_to_buy_or_sell = price * action - self.transaction_fee
##        self.balance += amount_to_buy_or_sell  # Updating the balance based on the trade
##
##    def calculate_reward(self):
##        daily_return = (self.balance / self.initial_balance) - 1
##
##        # Calculate Sharpe ratio based on returns so far
##        if len(self.returns) < 2:
##            sharpe_ratio = 0.1
##        else:
##            sharpe_ratio = np.mean(self.returns) / np.std(self.returns)
##
##        uncertainty_adjustment = 1  # You may include an uncertainty metric here
##        reward = daily_return * uncertainty_adjustment + sharpe_ratio
##        return reward
##
##    def render(self):
##        print(f"Current Step: {self.current_step} out of {len(self.data)}")
##        print(f"Current Balance: {self.balance}")


####################################################    

# Modify BayesianRewardCalculator class definition
class BayesianRewardCalculator:
    def __init__(self, config, model, guide):
        self.config = config
        self.model = model
        self.guide = guide
        self.optimizer = optim.Adam(self.guide.parameters(), lr=0.001)
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        
    def calculate_reward(self, current_state: pd.DataFrame, next_state: pd.DataFrame):
        state_data = torch.tensor(current_state.values, dtype=torch.float32)
        next_state_data = torch.tensor(next_state.values, dtype=torch.float32)
        
        loss = self.svi.step(state_data, next_state_data)
        
        return -loss  # Assuming reward is negative of loss

    
class UncertaintySAC:
    def __init__(self):
        # Simulated policy and Q-network outputs
        self.action_mean = torch.tensor([0.5])
        self.action_log_std = torch.tensor([0.1])
        self.uncertainty = torch.tensor([0.2])

    def exploration_action(self, state, lambda_):
        noise = torch.randn_like(self.action_mean) * self.uncertainty * lambda_
        return torch.clamp(self.action_mean + noise, min=-1, max=1)

class BayesianSharpeRatio:
    def __init__(self, alpha=1.0, beta=1.0, learnable_lambda=0.1):
        self.alpha = alpha
        self.beta = beta
        self.learnable_lambda = learnable_lambda  # Initialize learnable lambda

    def update(self, daily_return, opportunity_cost):
        self.alpha += daily_return if daily_return > 0 else 0
        self.beta += (1 - daily_return) if daily_return <= opportunity_cost else 0

        # Update learnable_lambda (can be replaced with a more sophisticated method)
        self.learnable_lambda = self.alpha / (self.alpha + self.beta)

    def calculate_dynamic_reward(self, state, action, next_state):
        volatility = self.calculate_historical_volatility(state['past_data'])
        performance = self.calculate_portfolio_performance(state['past_actions'])
        weighted_performance = performance * (self.alpha / (self.alpha + self.beta))
        reward = weighted_performance / (volatility + 1e-5)
        return reward

    def calculate_portfolio_performance(self, past_actions):
        raw_performance = np.sum(past_actions)
        weighted_performance = raw_performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance

    def calculate_historical_volatility(self, past_data):
        raw_volatility = np.std(past_data) if len(past_data) > 0 else 0
        weighted_volatility = raw_volatility * (self.beta / (self.alpha + self.beta))
        return weighted_volatility

    def step(self, market_state):
        # Add your agents' logic here
        pass

class BayesianSharpeRatio:
    def __init__(self, alpha=1.0, beta=1.0, learnable_lambda=0.1):
        self.alpha = alpha
        self.beta = beta
        self.learnable_lambda = learnable_lambda  # Initialize learnable lambda


    def update(self, daily_return):
        self.alpha += daily_return if daily_return > 0.0 else 0.0
        self.beta += 1 - daily_return if daily_return <= 0.0 else 0.0

        # Update learnable_lambda (can be replaced with a more sophisticated method)
        self.learnable_lambda = self.alpha / (self.alpha + self.beta)

    def sample(self, num_samples):
        return beta.rvs(self.alpha, self.beta, size=num_samples)
    
    def calculate_kelly_bet(self):
        mean_return = self.alpha / (self.alpha + self.beta)
        variance_return = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2.0 * (self.alpha + self.beta + 1.0))
        
        p_win = self.alpha / (self.alpha + self.beta)
        K = (mean_return * p_win - (1.0 - p_win)) / mean_return

        return K

    def calculate_dynamic_reward(self, state, action, next_state):
        
        weighted_performance = self.alpha / (self.alpha + self.beta)
        return weighted_performance

    def calculate_portfolio_performance(self, past_actions):
        raw_performance = np.sum(past_actions)
        weighted_performance = raw_performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance

    def calculate_historical_volatility(self, past_data):
        raw_volatility = np.std(past_data) if len(past_data) > 0 else 0
        weighted_volatility = raw_volatility * (self.beta / (self.alpha + self.beta))
        return weighted_volatility

from scipy.stats import beta
import numpy as np


import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class BayesianSharpeRatio:
    def __init__(self, alpha=1.0, beta=1.0, learnable_lambda=0.1):
        self.alpha = alpha
        self.beta = beta
        self.learnable_lambda = learnable_lambda

    def update(self, daily_return, opportunity_cost=0.0):
        self.alpha += max(daily_return, 0)
        self.beta += max(1 - daily_return, opportunity_cost)
        
        self.learnable_lambda = self.alpha / (self.alpha + self.beta)
        logging.info(f"Updated learnable_lambda: {self.learnable_lambda}")

    def calculate_kelly_bet(self):
        kelly_bet = self.learnable_lambda - (1 - self.learnable_lambda)
        logging.info(f"Calculated Kelly Bet: {kelly_bet}")
        return kelly_bet

    def sample(self, num_samples):
        return np.random.beta(self.alpha, self.beta, num_samples)

    def calculate_dynamic_reward(self, state, action, next_state):
        volatility = np.std(state['past_data']) if len(state['past_data']) > 0 else 0
        performance = np.sum(state['past_actions'])
        weighted_performance = performance * self.learnable_lambda
        reward = weighted_performance / (volatility + 1e-5)
        logging.info(f"Calculated dynamic reward: {reward}")
        return reward

# Placeholder for the CryptoTradingEnv class


class CryptoTradingEnv:
    def __init__(self, data, initial_balance=1000.0, transaction_fee=0.001, start_step=4000):
        self.data = data
        self.start_step = start_step
        self.current_step = start_step
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.balance = initial_balance
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.bayesian_sharpe = BayesianSharpeRatio()
        self.optimal_bet_size = 0.0  # Init optimal bet size

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.start_step
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.initial_balance = 1000
        self.optimal_bet_size = 0.0
        return self.get_state()
    
    def calculate_reward(self):
        # Calculate reward code
##        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        
        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
        price = self.data.iloc[self.current_step].get(asset_name, None)
        
        current_value = self.balance + self.btc_quantity * price
##        historical_dates = 
##
##        for i in self.returns.items():
##            historical_values += 
        
##        for asset, amount in self.returns.items():
##            current_value += self.data[self.current_step, 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'] * amount

        daily_return = (current_value / self.initial_balance) - 1.0
        self.bayesian_sharpe.update(daily_return)
        
        sharpe_samples = self.bayesian_sharpe.sample(len(self.returns)) if len(self.returns) > 2 else np.mean(self.returns) / np.std(self.returns)
        sharpe_samples = self.bayesian_sharpe.sample(1000)
        
        expected_sharpe = np.mean(sharpe_samples)
        uncertainty = np.std(sharpe_samples)
    
        reward = daily_return * expected_sharpe * self.balance * uncertainty
        
        print('reward: ', reward)
        print('uncertainty: ', uncertainty)
        print('expected_sharpe: ', expected_sharpe)
##        print('sharpe_samples: ', sharpe_samples)
        print('optimal_bet_size: ', self.optimal_bet_size)
        
        return reward
    
    def execute_trade(self, scaled_action):
        if isinstance(scaled_action, torch.Tensor):
##            action = action.item()
            scaled_action = scaled_action.detach().numpy()
        print('scaled_action: ', scaled_action)

        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
        price = self.data.iloc[self.current_step].get(asset_name, None)
        
        if price is None:
            print(f"Column {asset_name} not found in DataFrame.")
            return

        logging.info(f'price = {price} | action = {scaled_action}')
##        print(scaled_action)

        action_distribution = mean / (1.0 + std)

        if self.balance >1:
            print('scaled_action update: ', scaled_action)
            if scaled_action.all() > 0.0:  
                max_buyable = min(self.balance / (price * (1.0 + self.transaction_fee)), self.balance)  
                actual_buy = scaled_action * max_buyable
                cost = actual_buy * price * (1.0 + self.transaction_fee)
                self.balance = max(0.0, self.balance - cost)
                self.btc_quantity += actual_buy
                print('actual_buy: ', actual_buy)
                
            elif scaled_action.all() < 0.0:  
                max_sellable = self.btc_quantity
                actual_sell = -scaled_action * max_sellable
                revenue = actual_sell * price * (1.0 - self.transaction_fee)
                self.balance += revenue
                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
                print('sell revenue: ', revenue)

            else:
                pass

        else:
            if scaled_action.all() < 0.0:  
                max_sellable = self.btc_quantity
                actual_sell = -scaled_action * max_sellable
                revenue = actual_sell * price * (1.0 - self.transaction_fee)
                self.balance += revenue
                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
                print('sell revenue: ', revenue)

            else:
                pass


    def get_state(self):
        print('self.current_step', self.current_step)        
        row = self.data.iloc[self.current_step]
        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
        return np.array([float(self.balance)] + float_values)
    


    def dynamic_bet_sizing(self):
        """Calculates the optimal bet size using Bayesian-Kelly methods"""
        prior_mean = 0  # p_mean with activation function in the main/before env.step, state is prior/self.alpha earlier
        prior_std = 1  # p_variance with action function in the main/before env.step, state is prior/self.beta earlier

        # Update this to be more specific to your needs
        likelihood_mean = np.mean(self.returns)
        likelihood_std = np.std(self.returns)

        # Bayesian Updating
        posterior_mean = (likelihood_std ** 2 * prior_mean + prior_std ** 2 * likelihood_mean) / (prior_std ** 2 + likelihood_std ** 2)

        # Estimate probability of winning based on self.returns
        p_win = len([x for x in self.returns if x > 0]) / len(self.returns) if len(self.returns) > 0 else 0.5  # Default to 0.5/p_sum with activation
                                                                                                               # Could later be used as long|short threshold
        # Kelly Criterion Calculation
        K = (posterior_mean * p_win - (1 - p_win)) / posterior_mean if posterior_mean != 0 else 0  # Avoid division by zero

        self.optimal_bet_size = K

    def step(self, action):
##        self.dynamic_bet_sizing()  # Calculate the optimal bet size dynamically
##        scaled_action = action * self.optimal_bet_size  # Scale the action by the optimal bet size

        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()

        self.execute_trade(action * self.optimal_bet_size)  # Scale action by optimal bet size
        
        self.current_step += 1
##        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()

##        self.execute_trade(action * self.optimal_bet_size)  # Scale action by optimal bet size
        print('optimal_bet_size: ', self.optimal_bet_size)
        
        if self.current_step >= len(self.data):
            return self.reset(), 0, True

##        self.execute_trade(action)
        
        reward = self.calculate_reward()
##        reward += self.bayesian_sharpe.calculate_dynamic_reward
        print("Reward:", reward)  # Debugging line
        self.returns.append(reward.item() if torch.is_tensor(reward) else float(reward))
        
        done = self.current_step >= len(self.data) - 1
        
        return self.get_state(), reward, done
    
    def render(self):
        print(f"Current Step: {self.current_step} out of {len(self.data)}")
        print(f"Current Balance: {self.balance}")
        print(f"BTC quantity Owned:  {self.btc_quantity}")
##        print(f"returns:  {self.returns}")
        
    def _calculate_reward(self):
        daily_return = (self.balance / self.initial_balance) - 1.0
        sharpe_ratio = 0.01 if len(self.returns) < 2 else np.mean(self.returns) / np.std(self.returns)
        print("Sharpe Ratio:", sharpe_ratio)  # Debugging line
        return daily_return + sharpe_ratio

    def mcmc_sample(self, x, num_samples=10):
        # Implement MCMC sampling logic here
        mcmc_samples = []
        for _ in range(num_samples):
            output, *_ = self.forward(x)
            mcmc_samples.append(output)
        return torch.stack(mcmc_samples)  

    def forward(self, x):
        # Sample weight and bias
        weight = dist.Normal(self.weight_mu, self.weight_logstd.exp()).rsample()
        bias = dist.Normal(self.bias_mu, self.bias_logstd.exp()).rsample()
        
        # Bayesian Inference
        output = torch.matmul(x, weight.t()) + bias
        
        # Dirichlet distribution for multi-agent scenario
        dirichlet_dist = dist.Dirichlet(self.dirichlet_param)
        
        # Mixture of Beta distributions for short-term agent
        beta1 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around 1
        beta2 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around -1
        
        mixture = 0.5 * (beta1.sample() - 1) + 0.5 * (beta2.sample() + 1)
        
        # Alpha/Beta calculations
        alpha, beta = self.alpha_beta_nn(x).split(1, dim=1)
        
        # For Log Std
        log_std_output = output * self.log_std.exp()
        
        return output, alpha, beta, log_std_output, dirichlet_dist, mixture

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

class CryptoTradingEnv:
    def __init__(self, data, initial_balance=1000.0, transaction_fee=0.001, start_step=4000):
        # Data validation
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data should be a Pandas DataFrame.")
        
        self.data = data
        self.start_step = start_step
        self.current_step = start_step
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.balance = initial_balance
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.bayesian_sharpe = BayesianSharpeRatio()
        self.optimal_bet_size = 0.0

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.start_step
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.optimal_bet_size = 0.0
        return self.get_state()
    
    def calculate_reward(self):
        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
        price = self.data.iloc[self.current_step].get(asset_name, None)
        
        if price is None:
            logging.warning(f"Price for {asset_name} not found at step {self.current_step}.")
            return 0
        
        current_value = self.balance + self.btc_quantity * price
        daily_return = (current_value / self.initial_balance) - 1.0
        self.bayesian_sharpe.update(daily_return)
        
        sharpe_samples = self.bayesian_sharpe.sample(1000)
        expected_sharpe = np.mean(sharpe_samples)
        uncertainty = np.std(sharpe_samples)
    
        reward = daily_return * expected_sharpe * self.balance * uncertainty
        logging.info(f"Reward calculated: {reward}")
        
        return reward

    def step(self, action):
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        # Execute trade (to be implemented)
        self.execute_trade(action * self.optimal_bet_size)
        
        self.current_step += 1
        if self.current_step >= len(self.data):
            return self.reset(), 0, True
        
        reward = self.calculate_reward()
        self.returns.append(reward)
        
        done = self.current_step >= len(self.data) - 1
        return self.get_state(), reward, done

    def execute_trade(self, scaled_action):
        if isinstance(scaled_action, torch.Tensor):
##            action = action.item()
            scaled_action = scaled_action.detach().numpy()
        print('scaled_action: ', scaled_action)

        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
        price = self.data.iloc[self.current_step].get(asset_name, None)
        
        if price is None:
            print(f"Column {asset_name} not found in DataFrame.")
            return

        logging.info(f'price = {price} | action = {scaled_action}')
##        print(scaled_action)

##        action_distribution = mean / (1.0 + std)

        if self.balance >1:
            print('scaled_action update: ', scaled_action)
            if scaled_action.all() > 0.0:  
                max_buyable = min(self.balance / (price * (1.0 + self.transaction_fee)), self.balance)  
                actual_buy = scaled_action * max_buyable
                cost = actual_buy * price * (1.0 + self.transaction_fee)
                self.balance = max(0.0, self.balance - cost)
                self.btc_quantity += actual_buy
                print('actual_buy: ', actual_buy)
                
            elif scaled_action.all() < 0.0:  
                max_sellable = self.btc_quantity
                actual_sell = -scaled_action * max_sellable
                revenue = actual_sell * price * (1.0 - self.transaction_fee)
                self.balance += revenue
                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
                print('sell revenue: ', revenue)

            else:
                pass

        else:
            if scaled_action.all() < 0.0:  
                max_sellable = self.btc_quantity
                actual_sell = -scaled_action * max_sellable
                revenue = actual_sell * price * (1.0 - self.transaction_fee)
                self.balance += revenue
                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
                print('sell revenue: ', revenue)

            else:
                pass
        logging.info(f"Executed trade with action: {action}")

    def get_state(self):
        row = self.data.iloc[self.current_step]
        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
        return np.array([float(self.balance)] + float_values)
    
    def render(self):
        print(f"Current Step: {self.current_step} out of {len(self.data)}")
        print(f"Current Balance: {self.balance}")
        print(f"BTC quantity Owned:  {self.btc_quantity}")

    def _calculate_reward(self):
        daily_return = (self.balance / self.initial_balance) - 1.0
        sharpe_ratio = 0.01 if len(self.returns) < 2 else np.mean(self.returns) / np.std(self.returns)
        print("Sharpe Ratio:", sharpe_ratio)  # Debugging line
        return daily_return + sharpe_ratio

    def mcmc_sample(self, x, num_samples=10):
        # Implement MCMC sampling logic here
        mcmc_samples = []
        for _ in range(num_samples):
            output, *_ = self.forward(x)
            mcmc_samples.append(output)
        return torch.stack(mcmc_samples)  

    def forward(self, x):
        # Sample weight and bias
        weight = dist.Normal(self.weight_mu, self.weight_logstd.exp()).rsample()
        bias = dist.Normal(self.bias_mu, self.bias_logstd.exp()).rsample()
        
        # Bayesian Inference
        output = torch.matmul(x, weight.t()) + bias
        
        # Dirichlet distribution for multi-agent scenario
        dirichlet_dist = dist.Dirichlet(self.dirichlet_param)
        
        # Mixture of Beta distributions for short-term agent
        beta1 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around 1
        beta2 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around -1
        
        mixture = 0.5 * (beta1.sample() - 1) + 0.5 * (beta2.sample() + 1)
        
        # Alpha/Beta calculations
        alpha, beta = self.alpha_beta_nn(x).split(1, dim=1)
        
        # For Log Std
        log_std_output = output * self.log_std.exp()
        
        return output, alpha, beta, log_std_output, dirichlet_dist, mixture
##
##class CryptoTradingEnv:
##    def __init__(self, data, initial_balance=1000.0, transaction_fee=0.001, start_step=4000):
##        self.data = data
##        self.start_step = start_step
##        self.current_step = start_step
##        self.initial_balance = initial_balance
##        self.transaction_fee = transaction_fee
##        self.balance = initial_balance
##        self.returns = []
##        self.historical_values = []
##        self.btc_quantity = 0.0
##        self.bayesian_sharpe = BayesianSharpeRatio()
##        self.optimal_bet_size = 0.0
##
##    def reset(self):
##        self.balance = self.initial_balance
##        self.current_step = self.start_step
##        self.returns = []
##        self.historical_values = []
##        self.btc_quantity = 0.0
##        self.optimal_bet_size = 0.0
##        return self.get_state()
##
##    def execute_trade(self, scaled_action):
##        if isinstance(scaled_action, torch.Tensor):
####            action = action.item()
##            scaled_action = scaled_action.detach().numpy()
##        print('scaled_action: ', scaled_action)
##
##        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
##        price = self.data.iloc[self.current_step].get(asset_name, None)
##        
##        if price is None:
##            print(f"Column {asset_name} not found in DataFrame.")
##            return
##
##        logging.info(f'price = {price} | action = {scaled_action}')
####        print(scaled_action)
##
####        action_distribution = mean / (1.0 + std)
##
##        if self.balance >1:
##            print('scaled_action update: ', scaled_action)
##            if scaled_action.all() > 0.0:  
##                max_buyable = min(self.balance / (price * (1.0 + self.transaction_fee)), self.balance)  
##                actual_buy = scaled_action * max_buyable
##                cost = actual_buy * price * (1.0 + self.transaction_fee)
##                self.balance = max(0.0, self.balance - cost)
##                self.btc_quantity += actual_buy
##                print('actual_buy: ', actual_buy)
##                
##            elif scaled_action.all() < 0.0:  
##                max_sellable = self.btc_quantity
##                actual_sell = -scaled_action * max_sellable
##                revenue = actual_sell * price * (1.0 - self.transaction_fee)
##                self.balance += revenue
##                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
##                print('sell revenue: ', revenue)
##
##            else:
##                pass
##
##        else:
##            if scaled_action.all() < 0.0:  
##                max_sellable = self.btc_quantity
##                actual_sell = -scaled_action * max_sellable
##                revenue = actual_sell * price * (1.0 - self.transaction_fee)
##                self.balance += revenue
##                self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
##                print('sell revenue: ', revenue)
##
##            else:
##                pass
##        logging.info(f"Executed trade with action: {action}")
##
##    def calculate_reward(self):
##        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'
##        price = self.data.iloc[self.current_step].get(asset_name, None)
##        current_value = self.balance + self.btc_quantity * price
##        daily_return = (current_value / self.initial_balance) - 1.0
##
##        self.bayesian_sharpe.update(daily_return)
##        sharpe_samples = self.bayesian_sharpe.sample(1000)
##        expected_sharpe = np.mean(sharpe_samples)
##        uncertainty = np.std(sharpe_samples)
##        reward = daily_return * expected_sharpe * self.balance * uncertainty
##
##        logging.info(f"Calculated reward: {reward}")
##        return reward
##
##    def step(self, action):
##        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
##        self.execute_trade(action * self.optimal_bet_size)
##        self.current_step += 1
##
##        if self.current_step >= len(self.data):
##            return self.reset(), 0, True
##
##        reward = self.calculate_reward()
##        self.returns.append(reward)
##        done = self.current_step >= len(self.data) - 1
##
##        return self.get_state(), reward, done
##
##    def get_state(self):
##        row = self.data.iloc[self.current_step]
##        float_values = [float(x) for x in row.values]
##        return np.array([float(self.balance)] + float_values)
##
##    def render(self):
##        print(f"Current Step: {self.current_step} out of {len(self.data)}")
##        print(f"Current Balance: {self.balance}")
##        print(f"BTC quantity Owned:  {self.btc_quantity}")
##
##
##
##    def _calculate_reward(self):
##        daily_return = (self.balance / self.initial_balance) - 1.0
##        sharpe_ratio = 0.01 if len(self.returns) < 2 else np.mean(self.returns) / np.std(self.returns)
##        print("Sharpe Ratio:", sharpe_ratio)  # Debugging line
##        return daily_return + sharpe_ratio
##
##    def mcmc_sample(self, x, num_samples=10):
##        # Implement MCMC sampling logic here
##        mcmc_samples = []
##        for _ in range(num_samples):
##            output, *_ = self.forward(x)
##            mcmc_samples.append(output)
##        return torch.stack(mcmc_samples)  
##
##    def forward(self, x):
##        # Sample weight and bias
##        weight = dist.Normal(self.weight_mu, self.weight_logstd.exp()).rsample()
##        bias = dist.Normal(self.bias_mu, self.bias_logstd.exp()).rsample()
##        
##        # Bayesian Inference
##        output = torch.matmul(x, weight.t()) + bias
##        
##        # Dirichlet distribution for multi-agent scenario
##        dirichlet_dist = dist.Dirichlet(self.dirichlet_param)
##        
##        # Mixture of Beta distributions for short-term agent
##        beta1 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around 1
##        beta2 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around -1
##        
##        mixture = 0.5 * (beta1.sample() - 1) + 0.5 * (beta2.sample() + 1)
##        
##        # Alpha/Beta calculations
##        alpha, beta = self.alpha_beta_nn(x).split(1, dim=1)
##        
##        # For Log Std
##        log_std_output = output * self.log_std.exp()
##        
##        return output, alpha, beta, log_std_output, dirichlet_dist, mixture

# Placeholder for testing the integrated classes




##class CryptoTradingEnv:
##    def __init__(self, data, initial_balance=10000, transaction_fee=0.001, start_step=4000):#, config):
####        self.config = config
####        self.reward_calculator = BayesianRewardCalculator(self.config, your_model, your_guide)
##
##        self.data = data
##        self.start_step = start_step
##        self.current_step = start_step
##        self.initial_balance = initial_balance
##        self.transaction_fee = transaction_fee
##        self.balance = initial_balance
##        self.returns = []  # Initializing as a list
##
##    def reset(self):
##        self.balance = self.initial_balance
##        self.current_step = 4000
##        self.returns = []  # Resetting to an empty list
##        self.initial_balance = 10000
##        self.current_step = 4000
##        return self.get_state()
##
##    def get_state(self):
##        row = self.data.iloc[self.current_step]
##        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
##        return np.array([float(self.balance)] + float_values)
##
##
####    def step(self, actions):
####        # Your existing code
####        for idx, action in enumerate(actions):
####            self.execute_trade(idx, action)
####
####        # Update BayesianRewardCalculator
####        new_return = (self.balance / self.initial_balance) - 1
####        self.bayesian_reward_calculator.update(new_return)
####
####        reward = self.bayesian_reward_calculator.get_reward()
####        # Your existing code
##
##    def step(self, actions):
##        self.current_step += 1
##        if self.current_step >= len(self.data):
##            return self.reset(), 0, True
##    
##        for idx, action in enumerate(actions):
##            self.execute_trade(idx, action)
##
##        reward = self.calculate_reward()
##        self.returns.append(reward)
##        done = self.current_step >= len(self.data) - 1
##
##        return self.get_state(), reward, done
##
##
####        reward = self.reward_calculator.calculate_reward(self.current_state, self.next_state)
####        return next_state, reward, done, info
##
##
##
##    def execute_trade(self, asset_idx, action):
##        if isinstance(action, torch.Tensor):
##            action = action.detach().numpy()
##
##        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'  # Or use logic to find it dynamically
##
##        try:
##            price = self.data.iloc[self.current_step][asset_name]
##        except KeyError:
##            print(f"Column {asset_name} not found in DataFrame.")
##            return
##
##        logging.info(f'price = {price} | action = {action}')
##        amount_to_buy_or_sell = price * action - self.transaction_fee
##        self.balance += amount_to_buy_or_sell  # Updating the balance based on the trade
##
##    def calculate_reward(self):
##        daily_return = (self.balance / self.initial_balance) - 1
##
##        # Calculate Sharpe ratio based on returns so far
##        if len(self.returns) < 2:
##            sharpe_ratio = 0.1
##        else:
##            sharpe_ratio = np.mean(self.returns) / np.std(self.returns)
##
##        uncertainty_adjustment = 1  # You may include an uncertainty metric here
##        reward = daily_return * uncertainty_adjustment + sharpe_ratio
##        return reward
##
##    def render(self):
##        print(f"Current Step: {self.current_step} out of {len(self.data)}")
##        print(f"Current Balance: {self.balance}")


'''
Define actor
'''
##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim=2):
##        super(BayesianActor, self).__init__()
##        
##        # Define Linear layers
##        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
##        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
##        self.output = nn.Linear(hidden2_dim, output_dim)
##
##        # Initialize mean and std as learnable parameters
##        nn.init.xavier_uniform_(self.hidden1.weight)
##        nn.init.xavier_uniform_(self.hidden2.weight)
##        nn.init.xavier_uniform_(self.output.weight)
##        
##        self.hidden1_mean = nn.Parameter(self.hidden1.weight.clone())
##        self.hidden1_std = nn.Parameter(torch.zeros_like(self.hidden1.weight) + 0.05)
##        
##        self.hidden2_mean = nn.Parameter(self.hidden2.weight.clone())
##        self.hidden2_std = nn.Parameter(torch.zeros_like(self.hidden2.weight) + 0.05)
##        
##        self.output_mean = nn.Parameter(self.output.weight.clone())
##        self.output_std = nn.Parameter(torch.zeros_like(self.output.weight) + 0.05)
##        
##        self.to(device)
##
##    def forward(self, x):
##        # Sample weights and set them in nn.Linear layers
##        x.to(device)
##        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
##        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
##        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std))
##
##        x = F.selu(F.linear(x, weight=torch.normal(mean=self.hidden1_mean, std=self.hidden1_std), bias=self.hidden1.bias))
##        x = F.selu(F.linear(x, weight=torch.normal(mean=self.hidden2_mean, std=self.hidden2_std), bias=self.hidden2.bias))
##        x = F.linear(x, weight=torch.normal(mean=self.output_mean, std=self.output_std), bias=self.output.bias)
##
####        x = nn.Linear(weight=self.output_mean, std=self.output_std)
##        x = F.selu(F.linear(x, weight=hidden1_weight))
##        x = F.selu(F.linear(x, weight=hidden2_weight))
##        x = F.linear(x, weight=output_weight)
##        
##        
##        output_mean = x[:, 0]
##        output_std = F.softplus(x[:, 1])
##        print(f"output_std : {output_std}")
##        
##        output_std = torch.clamp(output_std, min=1e-3)
##        print(f"output_std : {output_std}")
##        print(f"output_mean: {output_mean}")
##        return output_mean, output_std



import torch
import torch.nn as nn
import torch.nn.functional as F


##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim=2):
##        super(BayesianActor, self).__init__()
##        
##        # Initialize mean and std as learnable parameters
##        self.hidden1_mean = nn.Parameter(torch.Tensor(hidden1_dim, input_dim))
##        self.hidden1_std = nn.Parameter(torch.Tensor(hidden1_dim, input_dim))
##        
##        self.hidden2_mean = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##        self.hidden2_std = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##        
##        self.output_mean = nn.Parameter(torch.Tensor(output_dim, hidden2_dim))
##        self.output_std = nn.Parameter(torch.Tensor(output_dim, hidden2_dim))
##
##        nn.init.xavier_uniform_(self.hidden1_mean)
##        nn.init.constant_(self.hidden1_std, 0.05)
##
##        nn.init.xavier_uniform_(self.hidden2_mean)
##        nn.init.constant_(self.hidden2_std, 0.05)
##
##        nn.init.xavier_uniform_(self.output_mean)
##        nn.init.constant_(self.output_std, 0.05)
##
##        self.to(device)
##
##    def forward(self, x):
##        x = x.to(device)
##        
##        # Sample weights from normal distribution
##        hidden1_weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std)).to(device)
##        hidden2_weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std)).to(device)
##        output_weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std)).to(device)
####        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
####        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
####        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std))
##
####        x = F.selu(F.linear(x, weight=torch.normal(mean=self.hidden1_mean, std=self.hidden1_std), bias=self.hidden1.bias))
####        x = F.selu(F.linear(x, weight=torch.normal(mean=self.hidden2_mean, std=self.hidden2_std), bias=self.hidden2.bias))
####        x = F.linear(x, weight=torch.normal(mean=self.output_mean, std=self.output_std), bias=self.output.bias)
##
##        # Perform forward pass
##        x = F.selu(F.linear(x, weight=hidden1_weight))
##        x = F.selu(F.linear(x, weight=hidden2_weight))
##        x = F.linear(x, weight=output_weight)
##        
##        output_mean = x[:, 0]
##        output_std = F.softplus(x[:, 1])
##        print(f"output_std : {output_std}")
##
##        output_std = torch.tanh(output_std)#, min=1e-3)
##        print(f"output_std : {output_std}")
##        print(f"output_mean: {output_mean}")        
##        return output_mean, output_std
##
######### Test the BayesianActor class

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet
import torch
import torch.nn as nn
import torch.nn.functional as F

##class BayesianLinear(nn.Module):
##    def __init__(self, in_features, out_features):
##        super(BayesianLinear, self).__init__()
##        self.in_features = in_features
##        self.out_features = out_features
##
##        # Mean and log variance parameters for weights
##        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
##        self.w_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-9, 0.1))
##
##        # Mean and log variance parameters for biases
##        self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
##        self.b_logvar = nn.Parameter(torch.Tensor(out_features).normal_(-9, 0.1))
##
##    def forward(self, x):
##        # Sample weights and biases from their posterior distributions
##        w_eps = torch.randn_like(self.w_mu)
##        w = self.w_mu + torch.exp(0.5 * self.w_logvar) * w_eps
##
##        b_eps = torch.randn_like(self.b_mu)
##        b = self.b_mu + torch.exp(0.5 * self.b_logvar) * b_eps
##
##        return F.linear(x, w, b)
##
##    def kl_divergence(self):
##        # Compute the KL divergence between the posterior and prior for both weights and biases
##        kl_w = -0.5 * torch.sum(1 + self.w_logvar - self.w_mu.pow(2) - self.w_logvar.exp())
##        kl_b = -0.5 * torch.sum(1 + self.b_logvar - self.b_mu.pow(2) - self.b_logvar.exp())
##        return kl_w + kl_b
##
### ELBO Loss
##def elbo_loss(output, target, kl_divergence):
##    likelihood = F.mse_loss(output, target, reduction='sum')
##    return likelihood + kl_divergence
##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, action_dim, hidden_dim=256):
##        super(BayesianActor, self).__init__()
##        self.fc1 = BayesianLinear(input_dim, hidden_dim)
##        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
##        self.mean_head = BayesianLinear(hidden_dim, action_dim)
##        self.log_std_head = BayesianLinear(hidden_dim, action_dim)
##        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
##        self.dirichlet_alpha = nn.Parameter(torch.ones(action_dim, requires_grad=True))
##    
##    def forward(self, state):
##        x = F.relu(self.fc1(state))
##        x = F.relu(self.fc2(x))
##        mean = self.mean_head(x)
##        log_std = self.log_std_head(x)
##        log_std = torch.clamp(log_std, min=-20, max=2)
##        return mean, log_std
##    
##    def sample(self, state):
##        mean, log_std = self.forward(state)
##        std = log_std.exp()
##        normal = Normal(mean, std)
##        z = normal.rsample()
##        action = torch.tanh(z)
##        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
##        log_prob = log_prob.sum(1, keepdim=True)
##        return action, log_prob
##    
##    def mc_sample(self, state, num_samples=100):
##        actions = []
##        for _ in range(num_samples):
##            action, _ = self.sample(state)
##            actions.append(action)
##        return torch.stack(actions).mean(dim=0)
##    
##    def estimate_uncertainty(self, state):
##        _, log_std = self.forward(state)
##        uncertainty = log_std.exp().mean().item()
##        return uncertainty
##    
##    def sample_dirichlet(self, state):
##        dirichlet = Dirichlet(self.dirichlet_alpha.exp())
##        sample = dirichlet.sample()
##        return sample


######if __name__ == '__main__':


##input_dim = 5
##hidden1_dim = 10
##hidden2_dim = 10
##output_dim = 2
##
##model = BayesianActor(input_dim, hidden1_dim, hidden2_dim, output_dim)
##x = torch.randn(3, input_dim)
##
##output_mean, output_std = model(x)
##print(f'output_mean: {output_mean}')
##print(f'output_std: {output_std}')

'''
Define Critic
'''
##class BayesianCritic(nn.Module):
##    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim):
##        super(BayesianCritic, self).__init__()
##
##        self.hidden1 = nn.Linear(state_dim + action_dim, hidden1_dim)
##        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
##        self.output = nn.Linear(hidden2_dim, 1)
##
##        # Initialize mean and std as learnable parameters
##        self.hidden1_mean = nn.Parameter(torch.Tensor(hidden1_dim, state_dim + action_dim))
##        self.hidden1_std = nn.Parameter(torch.Tensor(hidden1_dim, state_dim + action_dim))
##
##        self.hidden2_mean = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##        self.hidden2_std = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##
##        self.output_mean = nn.Parameter(torch.Tensor(1, hidden2_dim))
##        self.output_std = nn.Parameter(torch.Tensor(1, hidden2_dim))
##
##        # Xavier initialization for mean weights
##        nn.init.xavier_normal_(self.hidden1_mean)
##        nn.init.xavier_normal_(self.hidden2_mean)
##        nn.init.xavier_normal_(self.output_mean)
##
##        # Small constant initialization for std weights
##        nn.init.constant_(self.hidden1_std, 0.0001)
##        nn.init.constant_(self.hidden2_std, 0.0001)
##        nn.init.constant_(self.output_std, 0.0001)
##
##        self.to(device)
##
##    def forward(self, state, action):
##        state = state.to(device)
##        action = action.to(device)
##        
####        # Debugging: print shapes
##        print("State shape:", state.shape)
##        print("Action shape:", action.shape)
####
####        # Make sure action is a 2D tensor
####        if len(action.shape) == 1:
####            action = action.unsqueeze(-1)
####
####        # Concatenate state and action
####        x = torch.cat([state, action], dim=-1)
##        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)
##
##        # Sample weights (not modifying the class attributes)
####        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std).to(device)
####        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std).to(device)
####        output_weight = torch.normal(mean=self.output_mean, std=self.output_std).to(device)
####
####        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
####        x = F.selu(x)
####
####        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
####        x = F.selu(x)
####
####        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)
####
####        return q_value
##
##        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
##        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
##        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std))
##
##        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std).to(device)
##        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std).to(device)
##        output_weight = torch.normal(mean=self.output_mean, std=self.output_std).to(device)
##        
####        x = F.selu(self.hidden1(x))
####        x = F.selu(self.hidden2(x))
####        q_value = self.output(x)
####        q_value = self.output(x)
##
##        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
##        x = F.selu(x)
##
##        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
##        x = F.selu(x)
##
##        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)
##
##        return q_value

class BayesianLinear2(nn.Module):
    def __init__(self, in_features, out_features, n_actions=1):
        super().__init__()
        
        # Weight and bias priors
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logstd = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logstd = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        
        # For Dirichlet distribution
        self.dirichlet_param = nn.Parameter(torch.ones(n_actions))
        
        # For Alpha/Beta calculations
        self.alpha_beta_nn = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # For Log Std
        self.log_std = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Sample weight and bias
        weight = dist.Normal(self.weight_mu, self.weight_logstd.exp()).rsample()
        bias = dist.Normal(self.bias_mu, self.bias_logstd.exp()).rsample()
        
        # Bayesian Inference
        output = torch.matmul(x, weight.t()) + bias
        
        # Dirichlet distribution for multi-agent scenario
        dirichlet_dist = dist.Dirichlet(self.dirichlet_param)
        
        # Mixture of Beta distributions for short-term agent
        beta1 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around 1
        beta2 = dist.Beta(torch.tensor([2.0]), torch.tensor([2.0]))  # Centered around -1
        
        mixture = 0.5 * (beta1.rsample() - 1) + 0.5 * (beta2.rsample() + 1)
        
        # Alpha/Beta calculations
        alpha, beta = self.alpha_beta_nn(x).split(1, dim=1)
        
        # For Log Std
        log_std_output = output * self.log_std.exp()
        
        return output, alpha, beta, log_std_output, dirichlet_dist, mixture
    
    def kl_divergence(self, other):
        # Compute KL divergence between two BayesianLinear policies
        kl_div = dist.kl_divergence(
            dist.Normal(self.weight_mu, self.weight_logstd.exp()),
            dist.Normal(other.weight_mu, other.weight_logstd.exp())
        ).sum() + dist.kl_divergence(
            dist.Normal(self.bias_mu, self.bias_logstd.exp()),
            dist.Normal(other.bias_mu, other.bias_logstd.exp())
        ).sum()
        
        return kl_div

##class BayesianLinear1(nn.Module):
##    def __init__(self, in_features, out_features):
##        super(BayesianLinear1, self).__init__()
##        self.in_features = in_features
##        self.out_features = out_features
##        
##        # Initialize weights and biases
##        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
##        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
##        
##        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
##        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
##        
##        # Initialize hyperparameters for priors
##        self.weight_mu_prior = torch.Tensor([0.])
##        self.weight_sigma_prior = torch.Tensor([1.])
##        
##        self.bias_mu_prior = torch.Tensor([0.])
##        self.bias_sigma_prior = torch.Tensor([1.])
##        
##        self.softplus = nn.Softplus()
##        
##        # Initialize parameters
##        self.reset_parameters()
##        
##    def reset_parameters(self):
##        nn.init.xavier_normal_(self.weight_mu)
##        nn.init.xavier_normal_(self.weight_sigma)
##        nn.init.normal_(self.bias_mu, std=0.001)
##        nn.init.normal_(self.bias_sigma, std=0.001)
##        
##    def model(self):
##        weight = dist.Normal(self.weight_mu, self.softplus(self.weight_sigma))
##        bias = dist.Normal(self.bias_mu, self.softplus(self.bias_sigma))
##        
##        return weight, bias
##    
##    def guide(self):
##        weight = dist.Normal(self.weight_mu, self.softplus(self.weight_sigma))
##        bias = dist.Normal(self.bias_mu, self.softplus(self.bias_sigma))
##        
##        return weight, bias
##    
##    def elbo_loss(self, obs, pred):
##        weight, bias = self.model()
##        q_weight, q_bias = self.guide()
##        
##        # Log probabilities
##        log_p_w = weight.log_prob(self.weight_mu).sum()
##        log_p_b = bias.log_prob(self.bias_mu).sum()
##        
##        log_q_w = q_weight.log_prob(self.weight_mu).sum()
##        log_q_b = q_bias.log_prob(self.bias_mu).sum()
##        
##        mse_loss = F.mse_loss(pred, obs)
##        
##        return mse_loss - (log_p_w + log_p_b - log_q_w - log_q_b)
##        
##    def forward(self, x):
##        weight_sample = torch.normal(mean=self.weight_mu, std=self.softplus(self.weight_sigma))
##        bias_sample = torch.normal(mean=self.bias_mu, std=self.softplus(self.bias_sigma))
##        
##        return F.linear(x, weight_sample, bias_sample)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class BayesianLinear1(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5):
        super(BayesianLinear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples  # Number of MCMC samples
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize hyperparameters for priors
        self.weight_mu_prior = torch.Tensor([0.])
        self.weight_sigma_prior = torch.Tensor([1.])
        
        self.bias_mu_prior = torch.Tensor([0.])
        self.bias_sigma_prior = torch.Tensor([1.])
        
        self.softplus = nn.Softplus()
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.xavier_normal_(self.weight_sigma)
        nn.init.normal_(self.bias_mu, std=0.001)
        nn.init.normal_(self.bias_sigma, std=0.001)
        
    def model(self):
        weight = dist.Normal(self.weight_mu, self.softplus(self.weight_sigma))
        bias = dist.Normal(self.bias_mu, self.softplus(self.bias_sigma))
        
        return weight, bias
    
    def guide(self):
        weight = dist.Normal(self.weight_mu, self.softplus(self.weight_sigma))
        bias = dist.Normal(self.bias_mu, self.softplus(self.bias_sigma))
        
        return weight, bias
    
    def elbo_loss(self, obs, pred):
        weight, bias = self.model()
        q_weight, q_bias = self.guide()
        
        # Log probabilities
        log_p_w = weight.log_prob(self.weight_mu).sum()
        log_p_b = bias.log_prob(self.bias_mu).sum()
        
        log_q_w = q_weight.log_prob(self.weight_mu).sum()
        log_q_b = q_bias.log_prob(self.bias_mu).sum()
        
        mse_loss = F.mse_loss(pred, obs)
        
        return mse_loss - (log_p_w + log_p_b - log_q_w - log_q_b)
    
    def forward(self, x):
        predictions = []
        
        # MCMC sampling: average over multiple weight samples
        for _ in range(self.num_samples):
            weight_sample = torch.normal(mean=self.weight_mu, std=self.softplus(self.weight_sigma))
            bias_sample = torch.normal(mean=self.bias_mu, std=self.softplus(self.bias_sigma))
            
            pred = F.linear(x, weight_sample, bias_sample)
            predictions.append(pred)
        
        return torch.stack(predictions).mean(0)

    def kl_divergence(self, other):
        # Compute the KL divergence between this layer and another BayesianLinear1 layer
        mean1, std1 = self.mean, self.std
        mean2, std2 = other.mean, other.std

        kl = (torch.log(std2) - torch.log(std1)) + (std1**2 + (mean1 - mean2)**2) / (2 * std2**2) - 0.5
        return kl.sum()

    def update_parameters(self, new_mean, new_std):
        self.mean.data = new_mean.data
        self.std.data = new_std.data
        
    def sample_parameters(self):
        sampled_weight = torch.normal(mean=self.mean, std=self.std)
        return sampled_weight
'''    
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, n_actions=2):
        super().__init__()
        
        # Weight and bias priors
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logstd = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logstd = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        
        # For Dirichlet distribution
        self.dirichlet_param = nn.Parameter(torch.ones(n_actions))
        
        # For Alpha/Beta calculations
        self.alpha_beta_nn = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # For Log Std
        self.log_std = nn.Parameter(torch.zeros(1))
            
    def kl_divergence(self, other):
        # Compute KL divergence between two BayesianLinear policies
        kl_div = dist.kl_divergence(
            dist.Normal(self.weight_mu, self.weight_logstd.exp()),
            dist.Normal(other.weight_mu, other.weight_logstd.exp())
        ).sum() + dist.kl_divergence(
            dist.Normal(self.bias_mu, self.bias_logstd.exp()),
            dist.Normal(other.bias_mu, other.bias_logstd.exp())
        ).sum()
        
        return kl_div
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.xavier_normal_(self.weight_sigma)
        nn.init.normal_(self.bias_mu, std=0.001)
        nn.init.normal_(self.bias_sigma, std=0.001)
        
    def mcmc_sample(self, x, num_samples=10):
        # Implement MCMC sampling logic here
        mcmc_samples = []
        for _ in range(num_samples):
            output, *_ = self.forward(x)
            mcmc_samples.append(output)
        return torch.stack(mcmc_samples)
    
    @config_enumerate
    def model(self, x, y=None):
        # Define the probabilistic model for TraceEnum_ELBO
        weight = pyro.sample("weight", dist.Normal(self.weight_mu, self.weight_logstd.exp()))
        bias = pyro.sample("bias", dist.Normal(self.bias_mu, self.bias_logstd.exp()))
        
        mean = torch.matmul(x, weight.t()) + bias
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, 1), obs=y)
        return obs

    def guide(self, x, y=None):
        # Define the guide function for TraceEnum_ELBO
        weight = pyro.sample("weight", dist.Normal(self.weight_mu, self.weight_logstd.exp()))
        bias = pyro.sample("bias", dist.Normal(self.bias_mu, self.bias_logstd.exp()))
        
    def elbo(self, x, y):
        # Calculate TraceEnum_ELBO
        traceenum_elbo = TraceEnum_ELBO(max_plate_nesting=1)
        loss = traceenum_elbo.differentiable_loss(self.model, self.guide, x, y)
        return -loss  # return ELBO as a positive value
    
    def forward(self, x):
        predictions = []
        
        # MCMC sampling: average over multiple weight samples
        for _ in range(self.num_samples):
            weight_sample = torch.normal(mean=self.weight_mu, std=self.softplus(self.weight_sigma))
            bias_sample = torch.normal(mean=self.bias_mu, std=self.softplus(self.bias_sigma))
            
            pred = F.linear(x, weight_sample, bias_sample)
            predictions.append(pred)
        
        return torch.stack(predictions).mean(0)

    def update_parameters(self, new_mean, new_std):
        self.mean.data = new_mean.data
        self.std.data = new_std.data
        
    def sample_parameters(self):
        sampled_weight = torch.normal(mean=self.mean, std=self.std)
        return sampled_weight
    
'''    
'''
Certainly, a data pipeline would help clarify the flow of information through the system. Here's a simplified illustration:

1. **Data Ingestion**: Real-time financial data or historical data is fed into the system.
   - Input: Stock prices, trading volumes, etc.
  
2. **Data Preprocessing**: 
   - Features extraction, normalization, etc.
   - Output: Processed state variables

3. **Superstate Encoder** (Optional):
   - Input: Multiple state variables, potentially from multiple agents or times
   - Output: Encoded superstate

4. **Bayesian Actor**:
   - Input: Processed state variables or encoded superstate
   - Output: Action distribution parameters (mean, std)
  
5. **Action Sampling**:
   - Input: Action distribution parameters from Bayesian Actor
   - Output: Sampled action

6. **Environment Interaction**:
   - Input: Sampled action
   - Output: New state, reward, done flag

7. **Bayesian Reward Calculator**:
   - Input: New state, action
   - Output: Bayesian-calculated reward
  
8. **Bayesian Critic** (or other value function estimator):
   - Input: New state, Bayesian-calculated reward
   - Output: Value estimate
  
9. **Policy Update (TRPO, SAC, etc.)**:
   - Input: Value estimate, action distribution parameters
   - Output: Updated policy parameters

10. **Parameter Update for Bayesian Components**:
    - Input: Gradients calculated based on value estimate and Bayesian-calculated reward
    - Output: Updated Bayesian Actor and Bayesian Critic

11. **Logging and Monitoring**:
    - Storing important metrics, policy performance, etc.

Does this pipeline align with your vision, or are there elements you'd like to add or modify?
'''
    
'''
SAC agent
'''
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=3e-4, alpha=0.2, lr_alpha=1e-4):
        self.actor = BayesianActor(state_dim, hidden_dim, hidden_dim, 1).to(device)  # Mean and std
        self.critic1 = BayesianCritic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)
        self.critic2 = BayesianCritic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, dtype=torch.float32).to(device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()

    def sac_objective(self, q_values, mean, std):

        policy_distribution = torch.distributions.Normal(mean, std)

        unscaled_action = policy_distribution.rsample()  # Sample action before scaling
        
        # Squash the action to be in [-1, 1]
        scaled_action = torch.tanh(unscaled_action)
        
        # Compute log probability, scaled for the tanh transformation
        log_prob = policy_distribution.log_prob(unscaled_action) - torch.log(1 - scaled_action.pow(3) + 1e-6)
        
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        
        entropy_term = self.alpha * log_prob
        
        return q_values - entropy_term, scaled_action


    def update(self, states, actions, rewards, next_states, dones):
        states, actions, rewards, next_states, dones = map(lambda x: torch.tensor(x, dtype=torch.float32).to(device), 
                                                            [states, actions, rewards, next_states, dones])

        # Compute the Q-values
        q_value1 = self.critic1(states, actions)
        q_value2 = self.critic2(states, actions)

        # Compute the value of the next states using the critics
        with torch.no_grad():
            next_state_mean, next_state_std = self.actor(next_states)
            next_policy_distribution = torch.distributions.Normal(next_state_mean, next_state_std)
            next_sample_action = torch.tanh(next_policy_distribution.rsample())  # Scaled between [-1, 1]
            # another action
            next_q_value1 = self.critic1(next_states, next_sample_action)
            next_q_value2 = self.critic2(next_states, next_sample_action)
            min_next_q_value = torch.min(next_q_value1, next_q_value2)
            target_q_value = rewards + (1 - dones) * 0.99 * min_next_q_value

        # Compute the critic losses
        critic1_loss = F.mse_loss(q_value1, target_q_value)
        critic2_loss = F.mse_loss(q_value2, target_q_value)

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Optimize the actor
        mean, std = self.actor(states)
        sac_objective_1, adjusted_actions = self.sac_objective(q_value1, mean, std)
        actor_loss = -sac_objective_1.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the alpha parameter
        mean, std = self.actor(states)
        policy_distribution = torch.distributions.Normal(mean, std)
        unscaled_action = policy_distribution.rsample()  # Sample action before scaling
        log_prob = policy_distribution.log_prob(unscaled_action) - torch.log(1 - torch.tanh(unscaled_action).pow(3) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        print(critic1_loss.item())
        print(critic2_loss.item())
        print(actor_loss.item())
        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()
        
   









class BayesianCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim):
        super(BayesianCritic, self).__init__()

        self.hidden1 = nn.Linear(state_dim + action_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, 1)

        # Initialize mean and std as learnable parameters
        self.hidden1_mean = nn.Parameter(torch.Tensor(hidden1_dim, state_dim + action_dim))
        self.hidden1_std = nn.Parameter(torch.Tensor(hidden1_dim, state_dim + action_dim))

        self.hidden2_mean = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
        self.hidden2_std = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))

        self.output_mean = nn.Parameter(torch.Tensor(1, hidden2_dim))
        self.output_std = nn.Parameter(torch.Tensor(1, hidden2_dim))

        # Xavier initialization for mean weights
        nn.init.xavier_normal_(self.hidden1_mean)
        nn.init.xavier_normal_(self.hidden2_mean)
        nn.init.xavier_normal_(self.output_mean)

        # Small constant initialization for std weights
        nn.init.constant_(self.hidden1_std, 0.0001)
        nn.init.constant_(self.hidden2_std, 0.0001)
        nn.init.constant_(self.output_std, 0.0001)

        self.to(device)

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)
        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std)).to(device)
        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std)).to(device)
        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std)).to(device)

        hidden1_weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std)).to(device)
        hidden2_weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std)).to(device)
        output_weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std)).to(device)
        
        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
        x = F.selu(x)

        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
        x = F.selu(x)

        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)

        return q_value
##class BayesianCritic(nn.Module):
##    def __init__(self, state_dim, action_dim):
##        super(BayesianCritic, self).__init__()
##        self.linear1 = BayesianLinear(state_dim + action_dim, 256, 3)
##        self.linear2 = BayesianLinear(256, 256, 3)
##        self.linear3 = BayesianLinear(256, 1, 3)
##
##    def forward(self, state, action):
##        x = torch.cat([state, action], dim=1)
##        x = F.relu(self.linear1(x))
##        x = F.relu(self.linear2(x))
##        x = self.linear3(x)
##        return x
##
##    def kl_divergence(self):
##        return self.linear1.kl_divergence() + self.linear2.kl_divergence() + self.linear3.kl_divergence()
##
### ELBO Loss with Cubic-3 for Gradient Ascent
##def elbo_loss_cubic3(output, target, kl_divergence, cubic_coeff=1.0):
##    likelihood = F.mse_loss(output, target, reduction='sum')
##    elbo = likelihood + kl_divergence
##    return cubic_coeff * elbo ** 3  # Cubic-3 for Gradient Ascent


import torch
import torch.nn as nn
import torch.nn.functional as F


##class BayesianActor(nn.Module):
##    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim):
##        super(BayesianActor, self).__init__()
##
##        self.hidden1 = nn.Linear(state_dim, hidden1_dim)
##        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
##        self.action_mean = nn.Linear(hidden2_dim, action_dim)
##        self.action_std = nn.Linear(hidden2_dim, action_dim)
##
##        self.hidden1_mean = nn.Parameter(torch.Tensor(hidden1_dim, state_dim))
##        self.hidden1_std = nn.Parameter(torch.Tensor(hidden1_dim, state_dim))
##
##        self.hidden2_mean = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##        self.hidden2_std = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
##
##        self.action_mean_mean = nn.Parameter(torch.Tensor(action_dim, hidden2_dim))
##        self.action_mean_std = nn.Parameter(torch.Tensor(action_dim, hidden2_dim))
##
##        self.action_std_mean = nn.Parameter(torch.Tensor(action_dim, hidden2_dim))
##        self.action_std_std = nn.Parameter(torch.Tensor(action_dim, hidden2_dim))
##
##        nn.init.xavier_normal_(self.hidden1_mean)
##        nn.init.xavier_normal_(self.hidden2_mean)
##        nn.init.xavier_normal_(self.action_mean_mean)
##        nn.init.xavier_normal_(self.action_std_mean)
##
##        nn.init.constant_(self.hidden1_std, 0.0001)
##        nn.init.constant_(self.hidden2_std, 0.0001)
##        nn.init.constant_(self.action_mean_std, 0.0001)
##        nn.init.constant_(self.action_std_std, 0.0001)
##
##        self.to(device)
##
##    def forward(self, state):
##        state = state.to(device)
##        
##        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std)).to(device)
##        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std)).to(device)
##        self.action_mean.weight = nn.Parameter(torch.normal(mean=self.action_mean_mean, std=self.action_mean_std)).to(device)
##        self.action_std.weight = nn.Parameter(torch.normal(mean=self.action_std_mean, std=self.action_std_std)).to(device)
##        
##        x = F.linear(input=state, weight=self.hidden1.weight, bias=self.hidden1.bias)
##        x = F.selu(x)
##
##        x = F.linear(input=x, weight=self.hidden2.weight, bias=self.hidden2.bias)
##        x = F.selu(x)
##
##        mean = self.action_mean(x)
##        std = F.softplus(self.action_std(x))  # Ensure std is positive
##
##        return mean, std
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as dist
from pyro.infer import TraceEnum_ELBO, config_enumerate
import pyro


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, n_actions=2):
        super().__init__()
        
        # Weight and bias priors
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_logstd = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_logstd = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        
        # For Dirichlet distribution
        self.dirichlet_param = nn.Parameter(torch.ones(n_actions))
        
        # For Alpha/Beta calculations
        self.alpha_beta_nn = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # For Log Std
        self.log_std = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Sample weight and bias
        weight =  nn.Parameter(dist.Normal(self.weight_mu, self.weight_logstd.exp()).rsample())
        bias =  nn.Parameter(dist.Normal(self.bias_mu, self.bias_logstd.exp()).rsample())
        
        # Bayesian Inference
        output =  nn.Parameter(torch.matmul(x, weight.t()) + bias)
        
        # Dirichlet distribution for multi-agent scenario
##        dirichlet_dist =  nn.Parameter(dist.Dirichlet(self.dirichlet_param))
        
        # Mixture of Beta distributions for short-term agent
##        beta1 =  nn.Parameter(dist.Beta(torch.tensor([2.0]), torch.tensor([2.0])))  # Centered around 1
##        beta2 =  nn.Parameter(dist.Beta(torch.tensor([2.0]), torch.tensor([2.0])))  # Centered around -1
##        
##        mixture = 0.5 * (beta1.rsample() - 1) + 0.5 * (beta2.rsample() + 1)
        
        # Alpha/Beta calculations
##        alpha, beta = self.alpha_beta_nn(x).split(1, dim=1)
        
        # For Log Std
##        log_std_output = output * self.log_std.exp()
        
##        return output, alpha, beta, log_std_output, dirichlet_dist, mixture
        return weight, bias
    
    def kl_divergence(self, other):
        # Compute KL divergence between two BayesianLinear policies
        kl_div = dist.kl_divergence(
            dist.Normal(self.weight_mu, self.weight_logstd.exp()),
            dist.Normal(other.weight_mu, other.weight_logstd.exp())
        ).sum() + dist.kl_divergence(
            dist.Normal(self.bias_mu, self.bias_logstd.exp()),
            dist.Normal(other.bias_mu, other.bias_logstd.exp())
        ).sum()
        
        return kl_div

    def mcmc_sample(self, x, num_samples=10):
        # Implement MCMC sampling logic here
        mcmc_samples = []
        for _ in range(num_samples):
            output, *_ = self.forward(x)
            mcmc_samples.append(output)
        return torch.stack(mcmc_samples)
    
    @config_enumerate
    def model(self, x, y=None):
        # Define the probabilistic model for TraceEnum_ELBO
        weight = pyro.sample("weight", dist.Normal(self.weight_mu, self.weight_logstd.exp()))
        bias = pyro.sample("bias", dist.Normal(self.bias_mu, self.bias_logstd.exp()))
        
        mean = torch.matmul(x, weight.t()) + bias
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, 1), obs=y)
        return obs

    def guide(self, x, y=None):
        # Define the guide function for TraceEnum_ELBO
        weight = pyro.sample("weight", dist.Normal(self.weight_mu, self.weight_logstd.exp()))
        bias = pyro.sample("bias", dist.Normal(self.bias_mu, self.bias_logstd.exp()))
        
    def elbo(self, x, y):
        # Calculate TraceEnum_ELBO
        traceenum_elbo = TraceEnum_ELBO(max_plate_nesting=1)
        loss = traceenum_elbo.differentiable_loss(self.model, self.guide, x, y)
        return -loss  # return ELBO as a positive value

   
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##class BayesianActor(nn.Module):
class BayesianActor(BayesianLinear):

    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim):
        super(BayesianActor, self).__init__(state_dim, hidden1_dim)

        self.num = hidden1_dim
        # Define the network layers
        self.hidden1 = BayesianLinear(state_dim, hidden1_dim)
        self.hidden2 = BayesianLinear(hidden1_dim, hidden2_dim)
        self.action_mean = BayesianLinear(hidden2_dim, action_dim)
        self.action_std = BayesianLinear(hidden2_dim, action_dim)

        # Define Bayesian parameters for each layer
        self.init_bayesian_parameters(hidden1_dim, state_dim, 'hidden1')
        self.init_bayesian_parameters(hidden2_dim, hidden1_dim, 'hidden2')
        self.init_bayesian_parameters(action_dim, hidden2_dim, 'action_mean')
        self.init_bayesian_parameters(action_dim, hidden2_dim, 'action_std')
        
        self.to(device)

##    def mc(self, state, means, stds):
##        means, stds = [], []
##        for _ in range(state):
##            mean, std = super().forward(state_dim)
##            
##            print("Append Action Mean:", mean)
##            print("Append Action std:", std)
##            means.append(mean)
##            stds.append(std)
##
##        mean = torch.mean(torch.stack(means), dim=0)
##        std = torch.mean(torch.stack(stds), dim=0)
##        
##        # Dynamic Action Scaling
##        scaled_mean = mean / (1.0 + std)
##        
##        return scaled_mean, std

    def mc(self, state, means, stds, num_samples=10):
        means, stds = [], []
        for _ in range(num_samples):
            mean, std = super().forward(state)
            means.append(mean)
            stds.append(std)
            
        # Stack the samples and compute the mean and standard deviation along the sample dimension
        mean = torch.mean(torch.stack(means), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        
        # If you plan to use Kelly's Criterion, you can adjust `mean` and `std` here
        
        return mean, std


    def init_bayesian_parameters(self, out_dim, in_dim, layer_name):
        
        mean = nn.Parameter(torch.Tensor(out_dim, in_dim))
        std = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        
        # Xavier Initialization for mean
        nn.init.xavier_normal_(mean)
        nn.init.xavier_normal_(std)
        
        # Small constant initialization for std
        nn.init.constant_(std, 0.0001)
        nn.init.constant_(mean, 0.0001)        
        
        setattr(self, f"{layer_name}_mean", mean)
        setattr(self, f"{layer_name}_std", std)
            
    def forward(self, state):
        state = state.to(device)
        
        
        x = self.bayesian_linear(state, self.hidden1, 'hidden1')
        x = F.selu(x)

        x = self.bayesian_linear(x, self.hidden2, 'hidden2')
        x = F.selu(x)

        mean = self.bayesian_linear(x, self.action_mean, 'action_mean')
        std = F.softplus(self.bayesian_linear(x, self.action_std, 'action_std'))  # Ensure std is positive

        mean, std = self.mc(state, mean, std)
##        std = self.mc(state, mean, std)

        return mean, std

##    def forward(self, state, mc_samples=hidden1_dim):
        
##        means, stds = [], []

##        for _ in range(hidden1_dim):
##            mean, std = super().forward(state)
##            
##            print("Append Action Mean:", mean)
##            print("Append Action std:", std)
##            means.append(mean)
##            stds.append(std)
##
##        mean = torch.mean(torch.stack(means), dim=0)
##        std = torch.mean(torch.stack(stds), dim=0)
##        
##        # Dynamic Action Scaling
##        scaled_mean = mean / (1.0 + std)
##        
##        return scaled_mean, std

    def bayesian_linear(self, x, layer, layer_name):
        mean = getattr(self, f"{layer_name}_mean")
        std = getattr(self, f"{layer_name}_std")

        weight = nn.Parameter(torch.normal(mean=mean, std=std)).to(device)
        return F.linear(input=x, weight=weight, bias=layer.bias_mu)



    

# Example usage
##state_dim = 128
##action_dim = 2
##hidden1_dim = 128
##hidden2_dim = 128
##
##actor = BayesianActor(state_dim, action_dim, hidden1_dim, hidden2_dim)
##state = torch.rand((1, state_dim))
##mean, std = actor(state)
##
##print("Action Mean:", mean)
##print("Action Std:", std)



'''
BayseianValue:
'''

class TRPO:
    def __init__(self, policy, value, learning_rate=0.01):
        self.policy = policy
        self.value = value
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)

    def sample_trajectory(self, env, state, T):
        states, actions, rewards = [], [], []
        for t in range(T):
            action_prob = self.policy(torch.FloatTensor(state))
            action_dist = Categorical(action_prob)
            action = action_dist.rsample()
            next_state, reward, done, _ = env.step(action.item())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break
        return states, actions, rewards

    def compute_advantages(self, rewards, states, gamma=0.99):
        state_values = self.value(torch.FloatTensor(states))
        advantages = [0]
        for i in reversed(range(len(rewards) - 1)):
            delta = rewards[i] + gamma * state_values[i + 1] - state_values[i]
            advantage = delta + gamma * advantages[-1]
            advantages.insert(0, advantage)
        return advantages

    def update_policy(self, states, actions, advantages):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        advantages = torch.FloatTensor(advantages)

        old_prob = self.policy(states).detach()
        old_action_prob = old_prob.gather(1, actions)
        loss = -torch.min(
            self.policy(states).gather(1, actions) / old_action_prob * advantages,
            torch.clamp(self.policy(states).gather(1, actions) / old_action_prob, 1 - 0.2, 1 + 0.2) * advantages,
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




class TRPOBayesianValue:
    def __init__(self, bayesian_policy, bayesian_value, environment, delta=0.01, gamma=0.99, lam=0.95):
        self.policy = bayesian_policy
        self.value_net = bayesian_value
        self.env = environment
        self.delta = delta
        self.gamma = gamma
        self.lam = lam
        self.optimizer = optim.Adam(self.policy.parameters())
        self.value_optimizer = Adam({"lr": 0.01})
        self.svi = SVI(self.value_net.model, self.value_net.guide, self.value_optimizer, loss=Trace_ELBO())

    def gather_trajectories(self):
        states, actions, rewards, next_states, dones, old_probs, state_values = [], [], [], [], [], [], []
        state = self.env.reset()
        done = False

        while not done:
            with torch.no_grad():
                action_prob_means = self.policy.model(state)
                action_prob = self.policy.sample(action_prob_means)
                action = np.random.choice(len(action_prob), p=action_prob.cpu().numpy())

                state_value_dist = self.value_net.model(state)
                state_value = state_value_dist.rsample()

            next_state, reward, done, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_probs.append(action_prob[action])
            state_values.append(state_value)

            state = next_state

        return states, actions, rewards, next_states, old_probs, dones, state_values

    def compute_returns(self, rewards, state_values, dones):
        returns = []
        G = state_values[-1] * (1 - dones[-1])  # Bootstrap from last state
        for reward, value, done in zip(reversed(rewards), reversed(state_values), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        return returns

    def estimate_advantage(self, returns, state_values):
        return [G - value for G, value in zip(returns, state_values)]

    def train(self, num_iterations=1000):
        for iteration in range(num_iterations):
            states, actions, rewards, _, _, dones, state_values = self.gather_trajectories()
            returns = self.compute_returns(rewards, state_values, dones)
            advantages = self.estimate_advantage(returns, state_values)

            # Bayesian value network update
            for _ in range(5):  # Train the Bayesian value network for 5 epochs per iteration
                for state, return_ in zip(states, returns):
                    state_value_dist = self.value_net.model(state)
##                    loss = -state_value_dist.log_prob(return_).mean()
                    loss = self.svi.step(state, return_)
                    # Help me add Pyro's SVI (Stochastic Variational Inference) or a similar method here
                    # For simplicity, a typical loss.backward() is shown
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # TRPO update
            self.trpo_update(states, actions, advantages)

            # Print training details
            print(f"Iteration {iteration + 1}, Average Return: {np.mean(rewards)}")

    # Add to TRPOBayesianValue class
    def update_value_function(self):
        # Placeholder: The actual TRPO update method, which would use the given conjugate gradient and Fisher vector product functions.
        states, actions, rewards, next_states, old_probs, dones, state_values = self.gather_trajectories()

        # Compute returns, let's assume we have a function to do this
        returns = self.compute_returns(rewards, self.gamma)

        # Compute value loss
        value_loss = self.compute_value_loss(returns, state_values)

        # Update the value network (you may use regular PyTorch optimizers here)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()



    # Add to TRPOBayesianValue class
    def compute_policy_loss(self, action_probs, old_probs, advantages):
        # Calculate surrogate loss
        ratio = action_probs / old_probs
        surrogate = ratio * advantages
        return -torch.mean(torch.min(surrogate, torch.clamp(ratio, 1-self.delta, 1+self.delta) * advantages))

    def compute_value_loss(self, returns, values):
        # Mean squared error loss
        return torch.mean((returns - values)**2)

    # Add to TRPOBayesianValue class
    def update_value_function(self):
        states, actions, rewards, next_states, old_probs, dones, state_values = self.gather_trajectories()

        # Compute returns, let's assume we have a function to do this
        returns = self.compute_returns(rewards, self.gamma)

        # Compute value loss
        value_loss = self.compute_value_loss(returns, state_values)

        # Update the value network (you may use regular PyTorch optimizers here)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


    def trpo_update(self, states, actions, advantages):
        pass




'''
BayseianValue:
'''
class TRPOBayesianValue:
    def __init__(self, input_dim, hidden_dim, output_dim, environment, delta=0.01, gamma=0.99, lam=0.95):
        self.policy = BayesianPolicyNetwork(input_dim, hidden_dim, output_dim)
        self.value_net = BayesianValueNetwork(input_dim, hidden_dim)  # Assuming a BayesianValueNetwork class exists
        self.env = environment
        self.delta = delta
        self.gamma = gamma
        self.lam = lam
        
        # Policy optimizer using Pyro's SVI
        self.policy_optimizer = Adam({"lr": 0.01})
        self.policy_svi = SVI(self.policy.model, self.policy.guide, self.policy_optimizer, loss=Trace_ELBO())
        
        # Value optimizer using Pyro's SVI
        self.value_optimizer = Adam({"lr": 0.01})
        self.value_svi = SVI(self.value_net.model, self.value_net.guide, self.value_optimizer, loss=Trace_ELBO())

    def train(self, num_iterations=1000):
        for iteration in range(num_iterations):
            states, actions, rewards, _, _, dones, state_values = self.gather_trajectories()
            returns = self.compute_returns(rewards, state_values, dones)
            advantages = self.estimate_advantage(returns, state_values)

            # Bayesian policy network update
            for _ in range(5):
                for state, action in zip(states, actions):
                    policy_loss = self.policy_svi.step(state, action)
                    
            # Bayesian value network update
            for _ in range(5):
                for state, return_ in zip(states, returns):
                    value_loss = self.value_svi.step(state, return_)

            # TRPO update
            self.trpo_update(states, actions, advantages)

            # Print training details
            print(f"Iteration {iteration + 1}, Average Return: {np.mean(rewards)}")





'''
BayseianValueNet:
'''
class BayesianValueNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(BayesianValueNN, self).__init__()
        self.fc1 = BayesianLinear2(input_dim, hidden_dim)
        self.fc2 = BayesianLinear2(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear1(hidden_dim, 1)
        
    # Add this after our BayesianLinear definitions in __init__
##    self.tcn = nn.Sequential(
##        TCNBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=1),
##        TCNBlock(hidden_dim * 2, hidden_dim * 4, kernel_size=3, dilation=2),
##        # us can add more TCNBlocks here
##    )


    def forward(self, x):
        print(f"Shape of x at Layer L: {x.shape}")
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        print(f"Shape of x at Layer L+1: {x.shape}")
        x = self.fc3(x)
        # Adding Tanh activation to limit the output within [-1, 1]
        x = torch.tanh(x)
        
        # Adding an additional Gaussian distribution based on the network output
        action_mean = x
        action_std = F.softplus(self.action_log_std)  # Ensure standard deviation is positive

        return action_mean, action_std

    def model(self, x, y):
        fc1w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
        fc1b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
        
        fc2w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
        fc2b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))

        fc3w_prior = dist.Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight))
        fc3b_prior = dist.Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias))
        
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}

        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()

        lhat = lifted_reg_model(x)

        # Replace Bernoulli with Gaussian
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", Normal(lhat, 0.1), obs=y)
            
    def guide(self, x, y):
        fc1w_mu = torch.randn_like(self.fc1.weight)
        fc1w_sigma = torch.randn_like(self.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = pyro.param("fc1w_sigma", fc1w_sigma)
        fc1w_prior = dist.Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        
        fc1b_mu = torch.randn_like(self.fc1.bias)
        fc1b_sigma = torch.randn_like(self.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = pyro.param("fc1b_sigma", fc1b_sigma)
        fc1b_prior = dist.Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # For fc2
        fc2w_mu = torch.randn_like(self.fc2.weight)
        fc2w_sigma = torch.randn_like(self.fc2.weight)
        fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)

        fc2w_sigma_param = pyro.param("fc2w_sigma", fc2w_sigma)
        fc2w_prior = dist.Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
        
        fc2b_mu = torch.randn_like(self.fc2.bias)
        fc2b_sigma = torch.randn_like(self.fc2.bias)
        fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
        fc2b_sigma_param = pyro.param("fc2b_sigma", fc2b_sigma)
        fc2b_prior = dist.Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
        # And fc3 layers
        fc3w_mu = torch.randn_like(self.fc3.weight)
        fc3w_sigma = torch.randn_like(self.fc3.weight)
        fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
        fc3w_sigma_param = pyro.param("fc3w_sigma", fc3w_sigma)
        fc3w_prior = dist.Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
        
        fc3b_mu = torch.randn_like(self.fc3.bias)
        fc3b_sigma = torch.randn_like(self.fc3.bias)
        fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
        fc3b_sigma_param = pyro.param("fc3b_sigma", fc3b_sigma)
        fc3b_prior = dist.Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)
        
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, # Add fc2...
                  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, # And fc3...
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior
                  }
        
        lifted_module = pyro.random_module("module", self, priors)
        
        return lifted_module()

    def infer(self, x, y, num_iterations=1000):
        pyro.clear_param_store()
        optim = Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        for j in range(num_iterations):
            loss = svi.step(x, y)
            if j % 100 == 0:
                print(f"[iteration {j+1}] loss: {loss}")

    def BayesianSharpeRatio(num_samples=1000):
        portfolio_returns = get_portfolio_returns()  # This should return a tensor of shape [num_datapoints]
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns)
        
        # Bayesian inference for the mean and standard deviation of the returns
        mean_prior = dist.Normal(loc=0., scale=1.)
        std_prior = dist.Normal(loc=0., scale=1.)
        
        posterior_mean = pyro.sample('posterior_mean', dist.Normal(loc=mean_return, scale=0.1))
        posterior_std = pyro.sample('posterior_std', dist.Normal(loc=std_return, scale=0.1))
        
        sharpe_ratios = []
        
        for _ in range(num_samples):
            sampled_mean = pyro.sample('sampled_mean', dist.Normal(loc=posterior_mean, scale=0.1))
            sampled_std = pyro.sample('sampled_std', dist.Normal(loc=posterior_std, scale=0.1))
            
            sharpe_ratio = sampled_mean / sampled_std
            sharpe_ratios.append(sharpe_ratio)
            
        return torch.stack(sharpe_ratios).mean().item()

    def infer_action(bayesian_policy_nn, state, num_samples=100):
        sampled_actions = []
        for _ in range(num_samples):
            # This samples from the guide (posterior) and returns a 'new' model with those values
            sampled_nn = bayesian_policy_nn.guide(state, None)
            
            # Forward pass to get the action
            action_mean = sampled_nn(state)
            sampled_action = dist.Normal(action_mean, 0.1).rsample()
            sampled_actions.append(sampled_action)
            
        # Averaging over all the samples to get the final action
        final_action = torch.stack(sampled_actions).mean(0)
        
        return final_action


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random





# Define Network architectures
####class QNetwork(nn.Module):
##class BayesianCritic(nn.Module):
##    def __init__(self, input_dim, output_dim):
##        super(QNetwork, self).__init__()
##        self.fc = nn.Sequential(
##            nn.Linear(input_dim, 128),
##            nn.ReLU(),
##            nn.Linear(128, output_dim)
##        )
##
##    def forward(self, x):
##        return self.fc(x)
##
####class ActorNetwork(nn.Module):
##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, output_dim):
##        super(ActorNetwork, self).__init__()
##        self.fc = nn.Sequential(
##            nn.Linear(input_dim, 128),
##            nn.ReLU(),
##            nn.Linear(128, output_dim)
##        )
##
##    def forward(self, x):
##        return torch.tanh(self.fc(x))

### Initialize networks and optimizers
##state_dim = 4  # For example
##action_dim = 1  # For example
##Q1 = QNetwork(state_dim, action_dim)
##Q2 = QNetwork(state_dim, action_dim)
##Actor = ActorNetwork(state_dim, action_dim)
##optimizer_Q1 = optim.Adam(Q1.parameters())
##optimizer_Q2 = optim.Adam(Q2.parameters())
##optimizer_actor = optim.Adam(Actor.parameters())
##
### Hyperparameters
##gamma = 0.99
##alpha = 0.2
##tau = 0.005
##
### Replay Buffer
##replay_buffer = deque(maxlen=10000)

### Training Loop
##for episode in range(1000):  # Assuming 1000 episodes
##    state = np.array([0, 0, 0, 0])  # Get initial state from environment
##    done = False
##
##    while not done:
##        # Sample action from Gaussian policy
##        state_tensor = torch.FloatTensor(state).unsqueeze(0)
##        action = Actor(state_tensor).detach().numpy()[0]
##        
##        next_state = np.array([0, 0, 0, 0])  # Get next state from environment
##        reward = 0  # Get reward from environment
##        done = False  # Check if episode is done
##
##        # Store transition in replay buffer
##        replay_buffer.append((state, action, reward, next_state, done))
##
##        if len(replay_buffer) > 128:
##            # Sample mini-batch from replay buffer
##            mini_batch = random.sample(replay_buffer, 128)
##            states, actions, rewards, next_states, dones = zip(*mini_batch)
##            
##            # Convert to PyTorch tensors
##            state_tensor = torch.FloatTensor(states)
##            action_tensor = torch.FloatTensor(actions)
##            reward_tensor = torch.FloatTensor(rewards)
##            next_state_tensor = torch.FloatTensor(next_states)
##            done_tensor = torch.FloatTensor(dones)
##            
##            # Compute target Q-value
##            with torch.no_grad():
##                target = reward_tensor + gamma * torch.min(Q1(next_state_tensor), Q2(next_state_tensor)) * (1 - done_tensor)
##            
##            # Compute Q-value losses and backpropagate
##            Q1_loss = torch.pow(Q1(state_tensor) - target, 3).mean()
##            Q2_loss = torch.pow(Q2(state_tensor) - target, 3).mean()
##            optimizer_Q1.zero_grad()
##            optimizer_Q2.zero_grad()
##            Q1_loss.backward()
##            Q2_loss.backward()
##            optimizer_Q1.step()
##            optimizer_Q2.step()
##            
##            # Compute and backpropagate actor loss
##            policy_loss = -torch.abs(Q1(state_tensor)) + alpha * torch.log(torch.abs(action_tensor))
##            policy_loss = policy_loss.mean()
##            optimizer_actor.zero_grad()
##            policy_loss.backward()
##            optimizer_actor.step()
##
##        state = next_state




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
import pyro.distributions as dist

##class BayesianLinear(nn.Module):
##    def __init__(self, in_dim, out_dim):
##        super(BayesianLinear, self).__init__()
##        self.in_dim = in_dim
##        self.out_dim = out_dim
##        self.w_mu = nn.Parameter(torch.Tensor(out_dim, in_dim).normal_(0, 0.1))
##        self.w_rho = nn.Parameter(torch.Tensor(out_dim, in_dim).normal_(-3, 0.1))
##        self.b_mu = nn.Parameter(torch.Tensor(out_dim).normal_(0, 0.1))
##        self.b_rho = nn.Parameter(torch.Tensor(out_dim).normal_(-3, 0.1))
##
##    def forward(self, x):
##        w_eps = torch.randn_like(self.w_mu)
##        b_eps = torch.randn_like(self.b_mu)
##
##        w_sigma = torch.log1p(torch.exp(self.w_rho))
##        b_sigma = torch.log1p(torch.exp(self.b_rho))
##
##        w = self.w_mu + w_eps * w_sigma
##        b = self.b_mu + b_eps * b_sigma
##
##        return F.linear(x, w, b)
##
##
##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, output_dim, hidden_dim=128):
##        super(BayesianActor, self).__init__()
##        self.input_dim = input_dim
##        self.output_dim = output_dim
##        self.hidden_dim = hidden_dim
##
##        self.fc1 = BayesianLinear(input_dim, hidden_dim)
##        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
##        self.mu_head = BayesianLinear(hidden_dim, output_dim)  # Mean head
##        self.sigma_head = BayesianLinear(hidden_dim, output_dim)  # Std head
##
##    def forward(self, state):
##        x = F.selu(self.fc1(state))
##        x = F.selu(self.fc2(x))
##        mu = self.mu_head(x)
##        sigma = F.softplus(self.sigma_head(x))  # Make sure standard deviation is positive
##
##        return mu, sigma
##
##
##class BayesianCritic(nn.Module):
##    def __init__(self, input_dim, output_dim, hidden_dim=128):
##        super(BayesianCritic, self).__init__()
##        self.input_dim = input_dim
##        self.output_dim = output_dim
##        self.hidden_dim = hidden_dim
##
##        self.fc1 = BayesianLinear(input_dim, hidden_dim)
##        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
##        self.q_head = BayesianLinear(hidden_dim, output_dim)  # Q-value head
##
##    def forward(self, state, action):
##        x = torch.cat([state, action], dim=1)
##        x = F.selu(self.fc1(x))
##        x = F.selu(self.fc2(x))
##        q_value = self.q_head(x)
##
##        return q_value



'''
Driver
'''

from itertools import product

def main():
    # Hyperparameter space
    hyperparam_space = {
        'learning_rate': [1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        'entropy_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
        'hidden_dim': [128, 256, 512, 1024, 2048, 4096]
##        'transaction_fee': [0.001, 0.01],
    }
    
    # Generate all combinations of hyperparameters
    hyperparam_combinations = list(product(*hyperparam_space.values()))
    
    best_reward = -float('inf')
    best_hyperparams = None
    
    # Loop over all hyperparameter combinations
    for hyperparams in tqdm(hyperparam_combinations):
        
        learning_rate, entropy_weight, hidden_dim = hyperparams
        ### Parameters to tune ###
##        learning_rate = 3e-4
##        entropy_weight = 0.2
##        hidden_dim = 1024
        initial_balance =1000
        transaction_fee=0.005
        num_episodes = 500
        start_step = 4000
        ### Finished ###

        # Load and scale data
        data = pd.read_parquet('/Users/jiashengcheng/Desktop/Desktop - jashha 5 2 Mac Pro/Day_one/Test/cleaned_huopusa_7_1_23.parquet').ffill().fillna(0)
        data_scaled = data
        data_to_scale = data.drop(['timestamp','BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'], axis=1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_to_scale)
        data_scaled = pd.DataFrame(data_scaled, columns=data_to_scale.columns)
        data_scaled['timestamp'] = data['timestamp']
        data_scaled['BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'] = data['BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv']

        # Initialize environment and agent
        env = CryptoTradingEnv(data_scaled, initial_balance, transaction_fee, start_step)
        state_dim = len(env.get_state())
        action_dim = 1
        sac_agent = SACAgent(state_dim, action_dim, hidden_dim, learning_rate, entropy_weight)
        
        logging.info(f'state dim = {state_dim}')
        # Initialize environment and agent
        # Training loop
        n_episodes = num_episodes
        with tqdm(total=n_episodes) as pbar:
            for episode in tqdm(range(n_episodes)):
                pbar.set_postfix({"Episode": f"{episode+1}/{n_episodes}"}, refresh=True)
                
                state = env.reset()
                done = False
                episode_reward = 0.0
                actions = []
                while not done:
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    

                    # Get action
##                    with torch.no_grad():
####                        action_mean, action_std = sac_agent.actor(state_tensor)
####                        action_distribution = Normal(action_mean, action_std)
####                        sampled_action = action_distribution.rsample()
####                        tanh_action = torch.tanh(sampled_action)
##                        
##                        action_mean, action_std = sac_agent.actor(state_tensor)
##                        action_distribution = torch.distributions.Normal(action_mean, action_std)
##                        print('action is action_distribution :', action_distribution)
##                        sampled_action = action_distribution.rsample()
##                        print('action is after sampled_action :', sampled_action)
##
##                        action = torch.tanh(action_distribution.rsample()) 
##                        print('action is mains get action :', action)

                    with torch.no_grad():
                        action_output = sac_agent.actor(state_tensor)
                        print(f"Debug: action_output = {action_output}")

                        action_mean, action_std = action_output  # Assuming actor returns a tuple (mean, std)
                        print(f"Debug: action_mean = {action_mean}, action_std = {action_std}")

                        # Expand dimensions of action_std to match action_mean
                        expanded_action_std = action_std.unsqueeze(-1).expand_as(action_mean)

                        # Apply Softplus and clip to ensure std is positive
                        expanded_action_std = F.softplus(expanded_action_std).clamp(min=1e-5)

                        # Check shapes
                        print(f"Debug: Shapes - action_mean: {action_mean.shape}, expanded_action_std: {expanded_action_std.shape}")

                        action_distribution = torch.distributions.Normal(action_mean, expanded_action_std)
                        print(f"Debug: action_distribution = {action_distribution}")

##                        sampled_action = action_distribution.rsample()
##                        print(f"Debug: sampled_action = {sampled_action}")

                        # Convert to Dirichlet distribution parameters
                        dirichlet_params = torch.exp(action_mean)#, expanded_action_std)
                        dirichlet_distribution = torch.distributions.Dirichlet(dirichlet_params)

                        # Sample from Dirichlet and map to [-1, 1]
                        action = dirichlet_distribution.rsample()
                        action = 2 * action - 1  # Map to [-1, 1]
                        print('action is mains get action :', action)

                    actions.append(action)  # Print to see!!!
                    print('action is after mains get action :', action)
                    
        
                    # Take action in the environment
                    next_state, reward, done = env.step(action.cpu().numpy())
                    env.render()

                    # Update the current state and episode reward
                    state = next_state
                    episode_reward += reward


                    # Convert variables to tensors
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(device)
                    done_tensor = torch.FloatTensor([float(done)]).unsqueeze(0).to(device)
                    
                    # Update the agent
                    sac_agent.update(state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            
##                action_mean, action_std = sac_agent.actor(state_tensor)
##                action_std = torch.clamp(action_std, min=1e-3)
##                action_distribution = torch.distributions.Normal(action_mean, action_std)
##                action_distribution = action_mean / (1.0 + action_std)
                pbar.set_postfix({"Episode Reward": episode_reward}, refresh=True)

        # Compute episode reward, update best_hyperparams and best_reward if needed
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_hyperparams = hyperparams
            
    print(f'Best hyperparameters: {best_hyperparams}')
    print(f'Best reward: {best_reward}')
if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

######################################






######################################################

import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam




    
##class BayesianPolicyNN(nn.Module):
##    def __init__(self, state_dim, action_dim, hidden_dim):
##        super(BayesianPolicyNN, self).__init__()
##        
##        self.fc1 = nn.Linear(state_dim, hidden_dim)
##        self.fc2 = nn.Linear(hidden_dim, action_dim)
##        
##    def model(self, state, action):
##        # Prior distribution for network weights
##        fc1_w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
##        fc1_b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
##        
##        fc2_w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
##        fc2_b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))
##        
##        priors = {
##            'fc1.weight': fc1_w_prior, 'fc1.bias': fc1_b_prior,
##            'fc2.weight': fc2_w_prior, 'fc2.bias': fc2_b_prior
##        }
##        
##        # Lift the priors to enable sampling from them
##        lifted_module = pyro.random_module("module", self, priors)
##        
##        # Sample a neural network (which also samples w and b)
##        lifted_nn = lifted_module()
##        
##        # Run input data through the neural network
##        action_mean = lifted_nn(state)
##        
##        # Condition on the observed data
##        pyro.sample("obs", dist.Normal(action_mean, 0.1), obs=action)
##        
##    def guide(self, state, action):
##        # Variational parameters
##        fc1_w_mu = torch.randn_like(self.fc1.weight)
##        fc1_w_sigma = torch.randn_like(self.fc1.weight)
##        fc1_w_mu_param = pyro.param("fc1_w_mu", fc1_w_mu)
##        fc1_w_sigma_param = pyro.param("fc1_w_sigma", fc1_w_sigma, constraint=dist.constraints.positive)
##        epsilon = 1e-6
##        fc1_w_sigma_param = torch.clamp(fc1_w_sigma_param, min=epsilon)
##
##        fc1_b_mu = torch.randn_like(self.fc1.bias)
##        fc1_b_sigma = torch.randn_like(self.fc1.bias)
##        fc1_b_mu_param = pyro.param("fc1_b_mu", fc1_b_mu)
##        fc1_b_sigma_param = pyro.param("fc1_b_sigma", fc1_b_sigma, constraint=dist.constraints.positive)
##        
##        fc2_w_mu = torch.randn_like(self.fc2.weight)
##        fc2_w_sigma = torch.randn_like(self.fc2.weight)
##        fc2_w_mu_param = pyro.param("fc2_w_mu", fc2_w_mu)
##        
##        epsilon = 1e-6
##        fc1_w_sigma_param = torch.clamp(fc1_w_sigma_param, min=epsilon)
##
##    
##        fc2_w_sigma_param = pyro.param("fc2_w_sigma", fc2_w_sigma, constraint=dist.constraints.positive)
##        
##        fc2_b_mu = torch.randn_like(self.fc2.bias)
##        fc2_b_sigma = torch.randn_like(self.fc2.bias)
##        fc2_b_mu_param = pyro.param("fc2_b_mu", fc2_b_mu)
##        fc2_b_sigma_param = pyro.param("fc2_b_sigma", fc2_b_sigma, constraint=dist.constraints.positive)
##        
##        # Guide distributions
##        epsilon = 1e-6
##        fc1_w_dist = dist.Normal(loc=fc1_w_mu_param, scale=fc1_w_sigma_param + epsilon)
##        fc2_w_dist = dist.Normal(loc=fc1_w_mu_param, scale=fc1_w_sigma_param + epsilon)
##
####        fc1_w_dist = dist.Normal(loc=fc1_w_mu_param, scale=fc1_w_sigma_param)
##        fc1_b_dist = dist.Normal(loc=fc1_b_mu_param, scale=fc1_b_sigma_param)
##        
##        fc2_w_dist = dist.Normal(loc=fc2_w_mu_param, scale=fc2_w_sigma_param)
##        fc2_b_dist = dist.Normal(loc=fc2_b_mu_param, scale=fc2_b_sigma_param)
##
##
##        
##        dists = {
##            'fc1.weight': fc1_w_dist, 'fc1.bias': fc1_b_dist,
##            'fc2.weight': fc2_w_dist, 'fc2.bias': fc2_b_dist
##        }
##        
##        # Overloading the parameters in the neural network with random samples
##        lifted_module = pyro.random_module("module", self, dists)
##        
##        return lifted_module()
##


##### Initialize the Bayesian Neural Network
##state_dim = 24
##action_dim = 1
##hidden_dim = 48
##
##bayesian_policy_nn = BayesianPolicyNN(state_dim, action_dim, hidden_dim)
##
### Set up the optimizer and inference algorithm
##optimizer = Adam({"lr": 0.01})
##svi = SVI(bayesian_policy_nn.model, bayesian_policy_nn.guide, optimizer, loss=Trace_ELBO())
##
### Dummy data for testing
##state_data = torch.randn(100, state_dim)
##action_data = torch.randn(100, action_dim)
##
### Training loop
##num_iterations = 1000
##for j in range(num_iterations):
##    loss = svi.step(state_data, action_data)
##    if j % 100 == 0:
##        print(f"Epoch {j} Loss {loss}")





##############################################

### Example of how to use the infer_action function
##state = torch.FloatTensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)  # Replace this with a real state from your environment
##inferred_action = infer_action(bayesian_policy_nn, state)
##
##
##
##
##
##
### You can then call this function at the end of each episode to get the Bayesian Sharpe Ratio
##bayesian_sharpe_ratio = BayesianSharpeRatio()


##
##    def model(self, x, y):
##        fc1w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
##        fc1b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
##        
##        fc2w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
##        fc2b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))
##
##        fc3w_prior = dist.Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight))
##        fc3b_prior = dist.Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias))
##        
##        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
##                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}
##
##        lifted_module = pyro.random_module("module", self, priors)
##        lifted_reg_model = lifted_module()
##
##        lhat = lifted_reg_model(x)
##
##        # Replace Bernoulli with Gaussian
##        with pyro.plate("data", x.shape[0]):
##            pyro.sample("obs", Normal(lhat, 0.1), obs=y)
##
##
##
##
##
##
##def guide(self, x, y):
##    # learnable parameters for first layer weight
##    fc1w_mu = pyro.param("fc1w_mu", torch.randn_like(self.fc1.weight))
##    fc1w_sigma = pyro.param("fc1w_sigma", torch.ones_like(self.fc1.weight), constraint=dist.constraints.positive)
##    # learnable parameters for first layer bias
##    fc1b_mu = pyro.param("fc1b_mu", torch.randn_like(self.fc1.bias))
##    fc1b_sigma = pyro.param("fc1b_sigma", torch.ones_like(self.fc1.bias), constraint=dist.constraints.positive)
##    
##    # learnable parameters for second layer weight
##    fc2w_mu = pyro.param("fc2w_mu", torch.randn_like(self.fc2.weight))
##    fc2w_sigma = pyro.param("fc2w_sigma", torch.ones_like(self.fc2.weight), constraint=dist.constraints.positive)
##    # learnable parameters for second layer bias
##    fc2b_mu = pyro.param("fc2b_mu", torch.randn_like(self.fc2.bias))
##    fc2b_sigma = pyro.param("fc2b_sigma", torch.ones_like(self.fc2.bias), constraint=dist.constraints.positive)
##    
##    # learnable parameters for third layer weight
##    fc3w_mu = pyro.param("fc3w_mu", torch.randn_like(self.fc3.weight))
##    fc3w_sigma = pyro.param("fc3w_sigma", torch.ones_like(self.fc3.weight), constraint=dist.constraints.positive)
##    # learnable parameters for third layer bias
##    fc3b_mu = pyro.param("fc3b_mu", torch.randn_like(self.fc3.bias))
##    fc3b_sigma = pyro.param("fc3b_sigma", torch.ones_like(self.fc3.bias), constraint=dist.constraints.positive)
##
##    # define the variational parameters using these learnable parameters
##    priors = {'fc1.weight': dist.Normal(fc1w_mu, fc1w_sigma),
##              'fc1.bias': dist.Normal(fc1b_mu, fc1b_sigma),
##              'fc2.weight': dist.Normal(fc2w_mu, fc2w_sigma),
##              'fc2.bias': dist.Normal(fc2b_mu, fc2b_sigma),
##              'fc3.weight': dist.Normal(fc3w_mu, fc3w_sigma),
##              'fc3.bias': dist.Normal(fc3b_mu, fc3b_sigma)}
##    
##    lifted_module = pyro.random_module("module", self, priors)
##    lifted_reg_model = lifted_module()
##    return lifted_reg_model(x)
##
##Absolutely, let's dig deep into the interaction between `BayesianPolicyNN` and `SACAgent`, focusing on Automatic Differentiation Variational Inference (ADVI) for policy optimization and additional enhancements.
##
##### Overview:
##
###### Automatic Differentiation Variational Inference (ADVI) in BayesianPolicyNN
##ADVI is an advanced inference method to approximate the posterior distribution. This approach allows us to train our Bayesian neural network efficiently by maximizing the Evidence Lower Bound (ELBO).
##
###### Interaction with SACAgent
##`BayesianPolicyNN` serves as the policy for the `SACAgent`. When the agent needs to take an action, it consults `BayesianPolicyNN`. The Bayesian aspect allows the agent to factor in uncertainty, which could be beneficial for exploration in the RL setup.
##
###### Enhancements
##1. Action-Value Quantification: Quantify how certain the agent is about the Q-value of a given state-action pair.
##2. Hyperparameter tuning: Use Bayesian Optimization for hyperparameter tuning.
##3. Parallelization: Implement parallel data collection to speed up the training.
##
##### Python Code
##
##Firstly, let's update `BayesianPolicyNN` to include ADVI:
##
##'''python
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##import pyro
##from pyro.infer import SVI, Trace_ELBO
##import pyro.distributions as dist
##import pyro.optim as pyro_optim
##
##class BayesianPolicyNN(nn.Module):
##    def __init__(self, input_dim, hidden_dim, output_dim):
##        super(BayesianPolicyNN, self).__init__()
##        self.layer1 = nn.Linear(input_dim, hidden_dim)
##        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
##        self.layer3 = nn.Linear(hidden_dim, output_dim)
##        
##    def model(self, x):
##        pyro.module("BayesianPolicyNN", self)
##        h = F.selu(self.layer1(x))
##        h = F.selu(self.layer2(h))
##        action_prob = F.softmax(self.layer3(h), dim=-1)
##        return pyro.sample("action", dist.Categorical(action_prob))
##        
##    def guide(self, x):
##        # Define variational parameters
##        # Implement your ADVI logic here.
##        pass
##
##optimizer = pyro_optim.Adam({"lr": 1e-2})
##svi = SVI(BayesianPolicyNN.model, BayesianPolicyNN.guide, optimizer, loss=Trace_ELBO())
##'''
##
##Now, let's make `SACAgent` interact with this Bayesian network:
##
##'''python
##class SACAgent:
##    def __init__(self, state_dim, action_dim, actor, critic1, critic2):
##        self.actor = actor # Here, actor is an instance of BayesianPolicyNN
##        self.critic1 = critic1
##        self.critic2 = critic2
##
##    def select_action(self, state):
##        state_tensor = torch.FloatTensor(state).unsqueeze(0)
##        sampled_action = self.actor.model(state_tensor).cpu().detach().numpy()
##        return sampled_action
##
##    # ... rest of the SACAgent methods
##'''
##
##### Iterative Enhancements:
##
##1. **Action-Value Quantification**: Introduce a separate Bayesian NN for approximating Q-values, allowing the SACAgent to better understand the uncertainty in its evaluations.
##   
##2. **Bayesian Optimization**: For hyperparameter tuning, use Bayesian optimization techniques that suit well with our Bayesian framework.
##
##3. **Parallelization**: Employ techniques like A3C (Asynchronous Advantage Actor-Critic) to gather data in parallel which can significantly speed up the learning process.
##
##Starting with this setup allows us to iteratively refine and introduce more advanced features, creating a more robust and efficient agent over time. Would you like to delve into any of these enhancements in particular?
##
##
##Great questions. Let's dissect them one by one.
##
##### Filling in the `guide` Logic for ADVI
##The `guide` function in Pyro provides the variational approximation to the posterior, and it's where you'd implement the ADVI logic. The simplest guide would have the same structure as the model, and it would map from inputs to variational parameters for each random variable in the model.
##
##Here's a basic guide implementation:
##'''python
##def guide(self, x):
##    w_mu = pyro.param("w_mu", torch.randn(1, self.hidden_dim))
##    w_sigma = pyro.param("w_sigma", torch.randn(1, self.hidden_dim), constraint=dist.constraints.positive)
##    b_mu = pyro.param("b_mu", torch.randn(self.hidden_dim))
##    b_sigma = pyro.param("b_sigma", torch.randn(self.hidden_dim), constraint=dist.constraints.positive)
##
##    # Approximate the distribution of weights with a Normal distribution
##    w = pyro.sample("weights", dist.Normal(w_mu, w_sigma))
##    b = pyro.sample("bias", dist.Normal(b_mu, b_sigma))
##    
##    h = F.selu(self.layer1(x))
##    h = F.selu(self.layer2(h))
##    action_prob = F.softmax(self.layer3(h), dim=-1)
##    return pyro.sample("action", dist.Categorical(action_prob))
##'''
##
##### Action-Value Quantification
##Action-Value Quantification would be a separate concern and generally not part of the `guide` function. In the context of Bayesian RL, it can be used to estimate how certain the agent is about its Q-values. Essentially, you could create a Bayesian neural network to approximate the Q-function, similar to how the `BayesianPolicyNN` approximates the policy.
##
##### INFERENCE
##You've rightly noticed that I omitted the part where we actually perform the inference. In Pyro, after defining the model and the guide, you'd usually do something like this:
##
##'''python
### Train the guide with ADVI
##for epoch in range(1000):
##    loss = svi.step(x)
##    if epoch % 100 == 0:
##        print(f"Epoch {epoch} Loss {loss}")
##'''
##
##### Missing Elements
##1. **Data Collection**: How the data gets generated and fed into the system for both training and inference is missing in this example.
##2. **Reward Mechanism**: The code snippets don't include how the rewards are used to update the Q-functions and subsequently the policy.
##3. **State and Action Spaces**: They need to be defined explicitly based on your specific problem.
##4. **Optimization Loop**: There's no optimization loop that ties all the components together.
##  
##Would you like to go deeper into any of these elements?
##
##
##Certainly! I understand you're looking for a highly comprehensive, theoretical, and production-grade Python code that fully fleshes out the Bayesian Actor-Critic architecture, along with the accompanying elements like SAC (Soft Actor-Critic), Bayesian Policy Neural Networks, and other components.
##
##Given that the objective is complex, I'll start by giving an elaborate overview, and then present the Python code for the `BayesianPolicyNN` and `SACAgent` classes.
##
##### Elaborative Overview:
##
###### BayesianPolicyNN
##This neural network serves as the policy network with a Bayesian twist. It's responsible for making decisions (actions) based on the state of the environment. Bayesian neural networks (BNNs) incorporate uncertainty into their predictions, which allows us to quantify the confidence of the decisions it makes. 
##
##The BayesianPolicyNN will use variational inference as a method to approximate the posterior distribution. We use Pyro for this.
##
###### SACAgent
##This class utilizes Soft Actor-Critic, an off-policy actor-critic deep reinforcement learning algorithm. SAC provides stability and robustness in terms of learning, and the objective function for the agent includes an entropy term to encourage exploration. 
##
##The SACAgent will have methods for taking actions based on the current policy, computing the SAC objective which includes the entropy term, and updating its own parameters (both the actor and the critics) based on the observed reward and next state.
##
###### Data Pipeline
##Data will flow through a sequence of steps: 
##- Collection of state and other metrics from the environment
##- Decision-making via the BayesianPolicyNN
##- Execution of action and observation of reward and next state
##- Updating the SACAgent
##- Optionally, back-testing or running through a simulation to validate the efficacy of the trading strategy.
##
##---
##
##### Python Code:
##
##Firstly, let's import the required libraries:
##
##'''python
##import torch
##import torch.nn as nn
##import torch.optim as optim
##import pyro
##import pyro.distributions as dist
##from pyro.infer import SVI, Trace_ELBO
##'''
##
###### BayesianPolicyNN
##'''python
##class BayesianPolicyNN(nn.Module):
##    def __init__(self, input_dim, hidden_dim, output_dim):
##        super(BayesianPolicyNN, self).__init__()
##        self.layer1 = nn.Linear(input_dim, hidden_dim)
##        self.layer2 = nn.Linear(hidden_dim, hidden_dim
##
##)
##        self.layer3 = nn.Linear(hidden_dim, output_dim)
##    
##    def model(self, x):
##        pyro.module("BayesianPolicyNN", self)
##        h = F.selu(self.layer1(x))
##        h = F.selu(self.layer2(h))
##        action_prob = F.softmax(self.layer3(h), dim=-1)
##        return pyro.sample("action", dist.Categorical(action_prob))
##'''
##
###### SACAgent
##'''python
##class SACAgent:
##    def __init__(self, state_dim, action_dim, actor, critic1, critic2):
##        self.actor = actor
##        self.critic1 = critic1
##        self.critic2 = critic2
##        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
##        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-4)
##        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-4)
##        self.state_dim = state_dim
##        self.action_dim = action_dim
##
##    def select_action(self, state):
##        state = torch.FloatTensor(state).unsqueeze(0).to(device)
##        mu, sigma = self.actor(state)
##        distribution = dist.Normal(mu, sigma)
##        action = distribution.rsample()
##        return action.cpu().detach().numpy().flatten()
##
##    def update_parameters(self, memory):
##        # Your custom SAC objectives and update steps here
##        pass
##'''
##
##### Elaborative Review:
##- **BayesianPolicyNN**: In this architecture, we utilized Pyro's probabilistic programming capabilities to define a Bayesian neural network. This helps us handle uncertainty better when making action decisions. It's worth considering the trade-off between complexity and performance, especially in a time-sensitive environment like trading.
##  
##- **SACAgent**: The Soft Actor-Critic algorithm is an off-policy method, so it can learn efficiently. It utilizes two critics to stabilize learning. The agents can be further enhanced by introducing more Bayesian aspects into the critics or by using ensemble methods for robustness.
##
##- **Data Pipeline**: Since trading environments are highly dynamic and potentially costly in terms of mistakes, the Bayesian approach adds a layer of safety by providing a measure of uncertainty. However, Bayesian methods are computationally expensive, which might make real-time trading a challenge.
##
##In summary, this architecture aims to balance efficiency, robustness, and the ability to quantify uncertainty, providing a foundation for a sophisticated trading system. Would you like to discuss any specific section in more detail?
##
##
##
##
##
##
##
##
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##import torch.optim as optim
##import pyro
##import pyro.distributions as dist
##
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
##class BayesianLinear(nn.Module):
##    def __init__(self, in_dim, out_dim):
##        super(BayesianLinear, self).__init__()
##        self.in_dim = in_dim
##        self.out_dim = out_dim
##        self.w_mu = nn.Parameter(torch.Tensor(out_dim, in_dim).normal_(0, 0.1))
##        self.w_rho = nn.Parameter(torch.Tensor(out_dim, in_dim).normal_(-3, 0.1))
##        self.b_mu = nn.Parameter(torch.Tensor(out_dim).normal_(0, 0.1))
##        self.b_rho = nn.Parameter(torch.Tensor(out_dim).normal_(-3, 0.1))
##
##    def forward(self, x):
##        w_eps = torch.randn_like(self.w_mu)
##        b_eps = torch.randn_like(self.b_mu)
##
##        w_sigma = torch.log1p(torch.exp(self.w_rho))
##        b_sigma = torch.log1p(torch.exp(self.b_rho))
##
##        w = self.w_mu + w_eps * w_sigma
##        b = self.b_mu + b_eps * b_sigma
##
##        return F.linear(x, w, b)
##
##
##class BayesianActor(nn.Module):
##    def __init__(self, input_dim, output_dim, hidden_dim=128):
##        super(BayesianActor, self).__init__()
##        self.input_dim = input_dim
##        self.output_dim = output_dim
##        self.hidden_dim = hidden_dim
##
##        self.fc1 = BayesianLinear(input_dim, hidden_dim)
##        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
##        self.mu_head = BayesianLinear(hidden_dim, output_dim)  # Mean head
##        self.sigma_head = BayesianLinear(hidden_dim, output_dim)  # Std head
##
##    def forward(self, state):
##        x = F.selu(self.fc1(state))
##        x = F.selu(self.fc2(x))
##        mu = self.mu_head(x)
##        sigma = F.softplus(self.sigma_head(x))  # Make sure standard deviation is positive
##
##        return mu, sigma
##
##
##class BayesianCritic(nn.Module):
##    def __init__(self, input_dim, output_dim, hidden_dim=128):
##        super(BayesianCritic, self).__init__()
##        self.input_dim = input_dim
##        self.output_dim = output_dim
##        self.hidden_dim = hidden_dim
##
##        self.fc1 = BayesianLinear(input_dim, hidden_dim)
##        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
##        self.q_head = BayesianLinear(hidden_dim, output_dim)  # Q-value head
##
##    def forward(self, state, action):
##        x = torch.cat([state, action], dim=1)
##        x = F.selu(self.fc1(x))
##        x = F.selu(self.fc2(x))
##        q_value = self.q_head(x)
##
##        return q_value
##
##
##
##
##
##Great, you've picked some advanced features that could significantly boost the robustness and performance of your Bayesian Actor-Critic framework. I'll incorporate these into the classes you've requested. Please note that adding these features will make the code more complex, so I'll break down the additional functionalities and explain them as clearly as possible.
##
##Let's start by implementing these features in sequence, beginning with the `BayesianActor` class. I'll incorporate:
##
##- Variable MC Sampling for uncertainty-based decision-making
##- Uncertainty-Driven Exploration
##- An ensemble approach for different financial metrics like Sharpe Ratio
##- Placeholder for depth-wise Bayesian convolutional layers (and a note for adding TCNs)
##- Online Bayesian Updating
##- L1 Regularization on Uncertainty
##- Dropout as a Bayesian Approximation
##- Batching MC samples for efficiency
##
##Here's how the `BayesianActor` class would look:
##
##'''python
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
class BayesianActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(BayesianActor, self).__init__()

        # Bayesian layers with priors
        # [ADD YOUR BAYESIAN LAYERS HERE similar to BayesianCritic]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Sharpe Ratio head
        self.sharpe_head = nn.Linear(hidden_dim, 1)
        
        # Placeholder for TCNs
        # [ADD YOUR TCN LAYER HERE]
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.to(device)

    def forward(self, state, num_samples=10):
        # Online Bayesian updating can be placed here
        # [UPDATE PRIORS HERE]
        
        state = state.to(device)
        
        # Variable MC Sampling
        mc_samples = []
        for _ in range(num_samples):
            x = self.fc1(state)
            x = F.selu(x)
            
            # Dropout as Bayesian Approximation
            x = self.dropout(x)
            
            action = self.fc2(x)
            sharpe_ratio = self.sharpe_head(x)
            
            mc_samples.append((action, sharpe_ratio))

        # Combine MC samples
        action, sharpe_ratio = zip(*mc_samples)
        action = torch.stack(action).mean(dim=0)
        sharpe_ratio = torch.stack(sharpe_ratio).mean(dim=0)
        
        # Uncertainty-Driven Exploration
        # [IMPLEMENT UNCERTAINTY DRIVEN EXPLORATION HERE]

        return action, sharpe_ratio
##
### [DEFINE YOUR LOSS FUNCTION WITH L1 REGULARIZATION HERE]
### [DEFINE YOUR TRAINING LOOP WITH BATCHING MC SAMPLES AND TraceEnum_ELBO HERE]
##'''
##
##This is just a skeletal structure for `BayesianActor`. I've marked areas where you can add Bayesian layers, dropout as Bayesian approximation, and online Bayesian updating, among other features.
##
##Would you like to proceed with the `BayesianValueNN` and `TRPOBayesianValue` classes next, or would you like to discuss any of the features in more depth?
##
##
##
##
##
##Absolutely, integrating those advanced features can enhance the performance and reliability of the Actor-Critic framework in a Bayesian setting. Let's modify the `BayesianActor` and `BayesianCritic` classes to include the suggested features. Below is how you might integrate these features:
##
##1. Learnable parameters and forward methods for uncertainty estimation.
##2. Uncertainty-Based Learning Rate Annealing.
##3. Dynamic Action Scaling based on uncertainty.
##4. MC Sampling for forward passes.
##
##Here's an example:
##
##'''python
##import torch.optim as optim
##
### BayesianActor with enhancements
class BayesianActor(nn.Module):
    # ... (Initialization stays the same)
    
    def forward(self, state, mc_samples=1):
        means, stds = [], []
        for _ in range(mc_samples):
            mean, std = super().forward(state)
            means.append(mean)
            stds.append(std)

        mean = torch.mean(torch.stack(means), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        
        # Dynamic Action Scaling
        scaled_mean = mean / (1.0 + std)
        
        return scaled_mean, std

    # Learning rate annealing based on uncertainty
    def lr_annealing(optimizer, std, annealing_factor=0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1.0 - annealing_factor * torch.mean(std).item())
        
# Usage Example
actor = BayesianActor(state_dim, action_dim, 64, 64).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
state = torch.Tensor([some_state]).to(device)

# Uncertainty-Based Learning Rate Annealing
mean, std = actor(state)
lr_annealing(actor_optimizer, std)

# MC Sampling
mc_samples = 10
mc_mean, mc_std = actor(state, mc_samples=mc_samples)
##
### Dynamic Action Scaling: already in the BayesianActor's forward method.
##
##This is a simplified example; you may want to include checks and further sophistications based on your specific needs.
##
##For the `BayesianCritic` class, you could implement similar enhancements for uncertainty estimation and learning rate annealing. You would then use the Monte Carlo (MC) sampled Q-values in the TRPO or SAC learning updates.
##
##Would you like to continue with the `BayesianValueNN` and `TRPOBayesianValue` classes next?













































class BayesianRewardCalculator:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def scale_data(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    def update_alpha_beta(self, daily_return):
        self.alpha += max(0, daily_return)
        self.beta += max(0, 1 - daily_return)
        
    def calculate_reward(self, market_data, actions):
        # Assume market_data is a DataFrame and actions is a dict with 'BTC' and 'SP500' keys
        # Scale market data
        scaled_data = self.scale_data(market_data)

        # Compute daily returns
        daily_return = np.mean(scaled_data.pct_change().dropna())

        # Update alpha and beta
        self.update_alpha_beta(daily_return)

        # Calculate Bayesian reward
        alpha_weight = self.alpha / (self.alpha + self.beta)
        bayesian_return = daily_return * alpha_weight

        return bayesian_return





# Generate synthetic market data
market_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2022-01-01', end='2022-01-10'),
    'BTC': [30000, 31000, 30500, 31500, 32000, 32500, 33000, 33500, 34000, 34500],
    'SP500': [4200, 4210, 4205, 4215, 4220, 4225, 4230, 4235, 4240, 4245]
}).set_index('timestamp')

# Generate synthetic actions
actions = {'BTC': 1, 'SP500': -1}  # Buy BTC, Sell SP500

# Initialize BayesianRewardCalculator
reward_calculator = BayesianRewardCalculator()

# Calculate Bayesian reward
reward = reward_calculator.calculate_reward(market_data, actions)
print(f"Bayesian Reward: {reward}")
