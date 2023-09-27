
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
import torch.nn.functional as F
from gym import spaces
import numpy as np
import pandas as pd
import gym
import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

##
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)

import numpy as np

class BayesianRewardCalculator:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.returns = []
        self.opportunity_cost = 0.01  # Set your opportunity cost here
        self.past_data = []
        self.past_actions = []

    def update_returns(self, daily_return):
        self.returns.append(daily_return)

    def update_past_data(self, data_point):
        self.past_data.append(data_point)

    def update_past_actions(self, action):
        self.past_actions.append(action)

    def update_with_opportunity_cost(self, daily_return):
        self.alpha += daily_return if daily_return > 0 else 0
        self.beta += (1 - daily_return) if daily_return <= self.opportunity_cost else 0

    def calculate_dynamic_reward(self):
        volatility = self.calculate_historical_volatility()
        performance = self.calculate_portfolio_performance()
        weighted_performance = performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance / (volatility + 1e-5)

    def calculate_portfolio_performance(self):
        raw_performance = np.sum(self.past_actions)
        weighted_performance = raw_performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance

    def calculate_historical_volatility(self):
        raw_volatility = np.std(self.past_data) if len(self.past_data) > 0 else 0
        weighted_volatility = raw_volatility * (self.beta / (self.alpha + self.beta))
        return weighted_volatility

    def _calculate_base_metrics(self):
        if not self.returns:
            return 0.0, 0.0, 1.0

        daily_return = np.mean(self.returns[-30:]) if len(self.returns) > 30 else np.mean(self.returns)
        expected_return = self.alpha / (self.alpha + self.beta)
        uncertainty = self.beta / (self.alpha + self.beta + 1)

        return daily_return, expected_return, uncertainty






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



class CryptoTradingEnv:
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001, start_step=4000):
        self.data = data
        self.start_step = start_step
        self.current_step = start_step
        self.portfolio_manager = BayesianPortfolioManager(initial_balance, transaction_fee)

    def reset(self):
        self.current_step = 4000
        return self.portfolio_manager.reset()

    def get_state(self):
        row = self.data.iloc[self.current_step]
        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
        return np.array([float(self.portfolio_manager.balance)] + float_values)

    def step(self, actions):
        self.current_step += 1
        if self.current_step >= len(self.data):
            return self.reset(), 0, True

        asset_name = 'BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv'  # Or use logic to find it dynamically

        try:
            price = self.data.iloc[self.current_step][asset_name]
        except KeyError:
            print(f"Column {asset_name} not found in DataFrame.")
            return

        self.portfolio_manager.execute_trade(price, actions)

        reward = self.portfolio_manager.calculate_reward()
        self.portfolio_manager.update_returns(reward)
        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done







class CryptoTradingEnv(gym.Env):
    def __init__(self, data_path):
        super(CryptoTradingEnv, self).__init__()
        self.df = pd.read_parquet(data_path)
        self.state_space = len(self.df.columns)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        
    def reset(self):
        self.current_step = 0
        return self.df.iloc[self.current_step].values
    
    def step(self, action):
        self.current_step += 1
        next_state = self.df.iloc[self.current_step].values
        reward = self.calculate_rewards(action, next_state)
        done = False if self.current_step < len(self.df) - 1 else True
        return next_state, reward, done, {}
    
    def calculate_rewards(self, action, next_state):
        reward_calculator = BayesianRewardCalculator()
        return reward_calculator.compute_reward(action, next_state)


    
class BayesianRewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class BayesianRewardCalculator:
    def __init__(self):
        self.model = BayesianRewardNetwork(input_dim=100)  # Assuming input dimension is 100
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def compute_reward(self, action_distribution, action, state):
        self.model.train()
        # Convert state, action, and action_distribution to tensor
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        action_dist_tensor = torch.FloatTensor(action_distribution)
        
        # Concatenate them to form the input to the BNN
        input_tensor = torch.cat((state_tensor, action_tensor, action_dist_tensor), dim=0)
        
        # Forward pass to get reward
        bayesian_reward = self.model(input_tensor)
        
        # Calculate loss (Negative Trace_ELBO for example) and backpropagate
        loss = -torch.mean(torch.log(bayesian_reward))  # This is just a placeholder for actual loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return bayesian_reward.item()
class BayesianLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLayer, self).__init__()
        
        # Initialize mean and log variance parameters for the weights
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.2))
        self.weight_logvar = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(-5, 0.2))
        
        # Initialize mean and log variance parameters for the biases
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.2))
        self.bias_logvar = nn.Parameter(torch.Tensor(output_dim).normal_(-5, 0.2))
        
    def forward(self, x):
        # Sample weights and biases from their respective distributions
        weight_sample = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_logvar)
        bias_sample = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * torch.randn_like(self.bias_logvar)
        
        # Standard linear layer calculation with sampled weights and biases
        output = F.linear(x, weight_sample, bias_sample)
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal

##class BayesianTRPOPolicyNNnn(nn.Module):
##    def __init__(self, input_dim, hidden_dim, output_dim):
##        super(BayesianTRPOPolicyNNnn, self).__init__()
##        # Define linear layers
##        self.fc1 = nn.Linear(input_dim, hidden_dim)
##        self.fc2 = nn.Linear(hidden_dim, output_dim)
##        
##        # Initialize learnable mean and standard deviation for weights and biases of both layers
##        self.mean1 = nn.Parameter(torch.zeros(self.fc1.weight.shape))
##        self.std1 = nn.Parameter(torch.ones(self.fc1.weight.shape))
##        self.bias_mean1 = nn.Parameter(torch.zeros(self.fc1.bias.shape))
##        self.bias_std1 = nn.Parameter(torch.ones(self.fc1.bias.shape))
##        
##        self.mean2 = nn.Parameter(torch.zeros(self.fc2.weight.shape))
##        self.std2 = nn.Parameter(torch.ones(self.fc2.weight.shape))
##        self.bias_mean2 = nn.Parameter(torch.zeros(self.fc2.bias.shape))
##        self.bias_std2 = nn.Parameter(torch.ones(self.fc2.bias.shape))
##
##    def model(self, state):
##        mean1 = torch.zeros_like(self.fc1.weight)
##        std1 = torch.ones_like(self.fc1.weight)
##        bias_mean1 = torch.zeros_like(self.fc1.bias)
##        bias_std1 = torch.ones_like(self.fc1.bias)
##        
##        mean2 = torch.zeros_like(self.fc2.weight)
##        std2 = torch.ones_like(self.fc2.weight)
##        bias_mean2 = torch.zeros_like(self.fc2.bias)
##        bias_std2 = torch.ones_like(self.fc2.bias)
##
##        w1 = pyro.sample("w1", Normal(mean1, std1))
##        b1 = pyro.sample("b1", Normal(bias_mean1, bias_std1))
##        
##        w2 = pyro.sample("w2", Normal(mean2, std2))
##        b2 = pyro.sample("b2", Normal(bias_mean2, bias_std2))
##
##        x = F.relu(torch.matmul(state, w1.t()) + b1)
##        action_mean = torch.matmul(x, w2.t()) + b2
##        action_std = torch.exp(action_mean)  # Logarithm of variance to ensure positivity
##
##        return action_mean, action_std
##
##    def guide(self, state):
##        # For simplification, using same priors as model
##        # In a real-world application, these would be learned
##        w1 = pyro.sample("w1", Normal(torch.zeros_like(self.fc1.weight), torch.ones_like(self.fc1.weight)))
##        b1 = pyro.sample("b1", Normal(torch.zeros_like(self.fc1.bias), torch.ones_like(self.fc1.bias)))
##        
##        w2 = pyro.sample("w2", Normal(torch.zeros_like(self.fc2.weight), torch.ones_like(self.fc2.weight)))
##        b2 = pyro.sample("b2", Normal(torch.zeros_like(self.fc2.bias), torch.ones_like(self.fc2.bias)))
##
##        x = F.relu(torch.matmul(state, w1.t()) + b1)
##        action_mean = torch.matmul(x, w2.t()) + b2
##        action_std = torch.exp(action_mean)  # Logarithm of variance to ensure positivity
##
##        return action_mean, action_std
##
##    def forward(self, x):
##        # Sample a guide
##        sampled_guide = self.guide(x, None)
##        
##        # Forward pass through the sampled guide
##        return F.softmax(sampled_guide(x), dim=-1)
##
##
##
##
##
### Example usage:
##state_dim = 4  # State dimension
##action_dim = 2  # Action dimension
##hidden_dim = 128  # Hidden layer dimension
##
##policy = BayesianTRPOPolicyNNnn(state_dim, hidden_dim, action_dim)
##
### Generate some fake data for state
##state = torch.randn(5, state_dim)  # Batch of 5 states
##
### Model & guide should produce actions based on the state
##action_mean, action_std = policy.model(state)
##print("Model output:", action_mean, action_std)
##
##action_mean, action_std = policy.guide(state)
##print("Guide output:", action_mean, action_std)

# Implementing the rest (like natural gradients, Fisher-vector product, etc.) would follow.


import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal

    

##            
##class BayesianTRPOPolicyN(nn.Module):
##    def __init__(self, input_dim, hidden_dim, output_dim):
##        super(BayesianTRPOPolicyN, self).__init__()
##        # Define linear layers
##        self.fc1 = nn.Linear(input_dim, hidden_dim)
##        self.fc2 = nn.Linear(hidden_dim, output_dim)
##        
##        # Initialize learnable mean and standard deviation for weights and biases of both layers
##        self.mean1 = nn.Parameter(torch.zeros(self.fc1.weight.shape))
##        self.std1 = nn.Parameter(torch.ones(self.fc1.weight.shape))
##        self.w1 = nn.Parameter(torch.zeros(self.fc1.bias.shape))
##        self.b1 = nn.Parameter(torch.ones(self.fc1.bias.shape))
##        
##        self.mean2 = nn.Parameter(torch.zeros(self.fc2.weight.shape))
##        self.std2 = nn.Parameter(torch.ones(self.fc2.weight.shape))
##        self.w2 = nn.Parameter(torch.zeros(self.fc2.bias.shape))
##        self.b2 = nn.Parameter(torch.ones(self.fc2.bias.shape))
##
##    def model(self, state):
##        if state.dim() == 1:
##            state = state.unsqueeze(0)  # Ensure it's a batch of states
##            mean1 = torch.zeros_like(self.fc1.weight)
##            
##        std1 = torch.ones_like(self.fc1.weight)
##        bias_mean1 = torch.zeros_like(self.fc1.bias)
##        bias_std1 = torch.ones_like(self.fc1.bias)
##        
##        mean2 = torch.zeros_like(self.fc2.weight)
##        std2 = torch.ones_like(self.fc2.weight)
##        bias_mean2 = torch.zeros_like(self.fc2.bias)
##        bias_std2 = torch.ones_like(self.fc2.bias)
##
##        w1 = pyro.sample("w1", Normal(mean1, std1))
##        b1 = pyro.sample("b1", Normal(bias_mean1, bias_std1))
##        
##        w2 = pyro.sample("w2", Normal(mean2, std2))
##        b2 = pyro.sample("b2", Normal(bias_mean2, bias_std2))
##
##        x = F.relu(torch.matmul(state, w1.t()) + b1)
##        action_mean = torch.matmul(x, w2.t()) + b2
##        action_std = torch.exp(action_mean)  # Logarithm of variance to ensure positivity
##
##        return action_mean, action_std
##
##    def guide(self, state):
##        if state.dim() == 1:
##            state = state.unsqueeze(0)  # Ensure it's a batch of states
##        # For simplification, using same priors as model
##        # In a real-world application, these would be learned
##        w1 = pyro.sample("w1", Normal(torch.zeros_like(self.fc1.weight), torch.ones_like(self.fc1.weight)))
##        b1 = pyro.sample("b1", Normal(torch.zeros_like(self.fc1.bias), torch.ones_like(self.fc1.bias)))
##        
##        w2 = pyro.sample("w2", Normal(torch.zeros_like(self.fc2.weight), torch.ones_like(self.fc2.weight)))
##        b2 = pyro.sample("b2", Normal(torch.zeros_like(self.fc2.bias), torch.ones_like(self.fc2.bias)))
##
##        x = F.relu(torch.matmul(state, w1.t()) + b1)
##        action_mean = torch.matmul(x, w2.t()) + b2
##        action_std = torch.exp(action_mean)  # Logarithm of variance to ensure positivity
##
##        return action_mean, action_std
##
##    def forward(self, x):
##        action_mean, action_std = self.guide(x)  # This now unpacks the tuple
##        softmaxed_action_mean = F.softmax(action_mean, dim=-1)  # Using the unpacked action_mean
##        return softmaxed_action_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal



import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import Normal
import pyro

class BayesianTRPOPolicyN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianTRPOPolicyN, self).__init__()
        
        # Define linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize learnable mean and standard deviation for weights and biases of both layers
        self.mean1 = nn.Parameter(torch.zeros(self.fc1.weight.shape))
        self.std1 = nn.Parameter(torch.ones(self.fc1.weight.shape))
        self.bias_mean1 = nn.Parameter(torch.zeros(self.fc1.bias.shape))
        self.bias_std1 = nn.Parameter(torch.ones(self.fc1.bias.shape))
        
        self.mean2 = nn.Parameter(torch.zeros(self.fc2.weight.shape))
        self.std2 = nn.Parameter(torch.ones(self.fc2.weight.shape))
        self.bias_mean2 = nn.Parameter(torch.zeros(self.fc2.bias.shape))
        self.bias_std2 = nn.Parameter(torch.ones(self.fc2.bias.shape))

    def model(self, state):
        state = torch.FloatTensor(state)  # Make sure it's a tensor
        if len(state) == 1:
            state = state.unsqueeze(0)
        
        w1 = pyro.sample("w1", Normal(self.mean1, self.std1))
        b1 = pyro.sample("b1", Normal(self.bias_mean1, self.bias_std1))
        w2 = pyro.sample("w2", Normal(self.mean2, self.std2))
        b2 = pyro.sample("b2", Normal(self.bias_mean2, self.bias_std2))
        
        x = F.relu(state.matmul(w1.t()) + b1)
        action_mean = x.matmul(w2.t()) + b2
        action_std = torch.exp(action_mean)

        return action_mean, action_std

    def guide(self, state):
        state = torch.FloatTensor(state)  # Make sure it's a tensor

        if len(state) == 1:
            state = state.unsqueeze(0)
            
        w1 = pyro.sample("w1", Normal(self.mean1, self.std1))
        b1 = pyro.sample("b1", Normal(self.bias_mean1, self.bias_std1))
        w2 = pyro.sample("w2", Normal(self.mean2, self.std2))
        b2 = pyro.sample("b2", Normal(self.bias_mean2, self.bias_std2))

        x = F.relu(state.matmul(w1.t()) + b1)
        action_mean = x.matmul(w2.t()) + b2
        action_std = torch.exp(action_mean)

        return action_mean, action_std

    def forward(self, state):
        action_mean, action_std = self.guide(state)
        softmaxed_action_mean = F.softmax(action_mean, dim=-1)
        return softmaxed_action_mean
    



    # ... (rest of the code)


    # ... (rest of the code)
### Example usage, replace dimensions as needed
##input_dim = 10
##hidden_dim = 20
##output_dim = 5
##
##policy = BayesianTRPOPolicyNN(input_dim, hidden_dim, output_dim)
##
##state = torch.rand((1, input_dim))  # Example state
##output = policy(state)
##
### Example usage:
##state_dim = 4  # State dimension
##action_dim = 2  # Action dimension
##hidden_dim = 128  # Hidden layer dimension
##
##policy = BayesianTRPOPolicyNN(state_dim, hidden_dim, action_dim)
##
### Generate some fake data for state
##state = torch.randn(5, state_dim)  # Batch of 5 states
##
### Model & guide should produce actions based on the state
##action_mean, action_std = policy.model(state)
##print("Model output:", action_mean, action_std)
##
##action_mean, action_std = policy.guide(state)
##print("Guide output:", action_mean, action_std)

# Implementing the rest (like natural gradients, Fisher-vector product, etc.) would follow.

class BayesianTRPOValue:
    def __init__(self, input_dim, hidden_dim, output_dim, environment, delta=0.01, gamma=0.99, lam=0.95):
        self.policy = BayesianTRPOPolicyN(input_dim, hidden_dim, output_dim)
        self.value_net = BayesianValueNN(input_dim, hidden_dim)#, output_dim, environment)  # Assuming a BayesianValueNetwork class exists
##        self.value_net = 
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

### You can then call this function at the end of each episode to get the Bayesian Sharpe Ratio
##bayesian_sharpe_ratio = BayesianSharpeRatio()


import numpy as np
from datetime import datetime

class BayesianRewardCalculator:
    def __init__(self, alpha=1.0, beta=1.0, initial_opportunity_cost=0.01):
        self.alpha = alpha
        self.beta = beta
        self.returns = []
        self.opportunity_cost = initial_opportunity_cost
        self.past_data = []
        self.past_actions = []
        self.holding_periods = {}  # To track the holding period for tax calculations

    def update_returns(self, daily_return):
        self.returns.append(daily_return)

    def update_past_data(self, data_point):
        self.past_data.append(data_point)

    def update_past_actions(self, action, timestamp):
        self.past_actions.append(action)
        self.holding_periods[timestamp] = datetime.now()  # Assuming the action corresponds to buying or selling an asset

    def update_with_opportunity_cost(self, daily_return):
        self.alpha += daily_return if daily_return > 0 else 0
        self.beta += (1 - daily_return) if daily_return <= self.opportunity_cost else 0

    def update_opportunity_cost(self, new_opportunity_cost):
        self.opportunity_cost = new_opportunity_cost

    def calculate_dynamic_reward(self):
        volatility = self.calculate_historical_volatility()
        performance = self.calculate_portfolio_performance()
        weighted_performance = performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance / (volatility + 1e-5)

    def calculate_portfolio_performance(self):
        raw_performance = np.sum(self.past_actions)
        weighted_performance = raw_performance * (self.alpha / (self.alpha + self.beta))
        return weighted_performance

    def calculate_historical_volatility(self):
        raw_volatility = np.std(self.past_data) if len(self.past_data) > 0 else 0
        weighted_volatility = raw_volatility * (self.beta / (self.alpha + self.beta))
        return weighted_volatility

    def calculate_sharpe_ratio(self):
        if len(self.returns) == 0:
            return 0.0

        average_return = np.mean(self.returns)
        standard_deviation = np.std(self.returns)
        if standard_deviation == 0:
            return 0.0
        
        return (average_return / standard_deviation) * np.sqrt(len(self.returns))

    def calculate_tax_implications(self, timestamp_now):
        tax = 0
        for timestamp, action_time in self.holding_periods.items():
            holding_period_months = (timestamp_now - action_time).days // 30  # Simplified calculation

            if holding_period_months > 8:
                tax += 0.1  # 10% tax for holding more than 8 months
            else:
                tax += 0.4  # 40% tax for holding less than 8 months

        return tax

    def _calculate_base_metrics(self):
        if not self.returns:
            return 0.0, 0.0, 1.0

        daily_return = np.mean(self.returns[-30:]) if len(self.returns) > 30 else np.mean(self.returns)
        expected_return = self.alpha / (self.alpha + self.beta)
        uncertainty = self.beta / (self.alpha + self.beta + 1)

        return daily_return, expected_return, uncertainty


import numpy as np
from datetime import datetime

class BayesianRewardCalculator:
    def __init__(self, alpha=1.0, beta=1.0, initial_opportunity_cost=0.01):
        self.alpha = alpha
        self.beta = beta
        self.returns = []
        self.opportunity_cost = initial_opportunity_cost
        self.short_term_actions = []
        self.long_term_actions = []
        self.short_term_returns = []
        self.long_term_returns = []
        self.short_term_holding_periods = {}
        self.long_term_holding_periods = {}
        
    def update_returns(self, daily_return, agent_type):
        self.returns.append(daily_return)
        if agent_type == 'short_term':
            self.short_term_returns.append(daily_return)
        else:
            self.long_term_returns.append(daily_return)

    def update_actions(self, action, timestamp, agent_type):
        if agent_type == 'short_term':
            self.short_term_actions.append(action)
            self.short_term_holding_periods[timestamp] = datetime.now()
        else:
            self.long_term_actions.append(action)
            self.long_term_holding_periods[timestamp] = datetime.now()

    def update_with_opportunity_cost(self, daily_return, agent_type):
        self.alpha += daily_return if daily_return > 0 else 0
        self.beta += (1 - daily_return) if daily_return <= self.opportunity_cost else 0
        
        if agent_type == 'short_term':
            self.short_term_returns.append(daily_return)
        else:
            self.long_term_returns.append(daily_return)

    def calculate_dynamic_reward(self, agent_type):
        actions = self.short_term_actions if agent_type == 'short_term' else self.long_term_actions
        returns = self.short_term_returns if agent_type == 'short_term' else self.long_term_returns

        performance = np.sum(actions) * (self.alpha / (self.alpha + self.beta))
        volatility = np.std(returns) if len(returns) > 0 else 0
        volatility *= (self.beta / (self.alpha + self.beta))
        
        return performance / (volatility + 1e-5)
    
    def settle_conflicts(self):
        # Your conflict resolution logic here, e.g., dividing available cash between short-term and long-term goals
        pass
    
    def calculate_tax_implications(self, timestamp_now, agent_type):
        holding_periods = self.short_term_holding_periods if agent_type == 'short_term' else self.long_term_holding_periods
        tax = 0
        for timestamp, action_time in holding_periods.items():
            holding_period_months = (timestamp_now - action_time).days // 30  # Simplified calculation
            tax += 0.1 if holding_period_months > 8 else 0.4
        return tax


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


# Add these imports at the beginning of your code
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
import torch.optim as optim

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


# Assuming your CryptoEnvironment looks somewhat like this
class CryptoEnvironment:
    def __init__(self, config):
        self.config = config
        self.reward_calculator = BayesianRewardCalculator(self.config, your_model, your_guide)
        
    def step(self, action):
        # Your logic to transition from current_state to next_state
        # ...
        
        reward = self.reward_calculator.calculate_reward(self.current_state, self.next_state)
        return next_state, reward, done, info





import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro.optim as pyro_optim

class BayesianPolicyNnnN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianPolicyNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def model(self, x):
        pyro.module("BayesianPolicyNN", self)
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        action_prob = F.softmax(self.layer3(h), dim=-1)
        return pyro.sample("action", dist.Categorical(action_prob))
        
        # Define variational parameters
        # Implement your ADVI logic here.
    def guide(self, x):
        w_mu = pyro.param("w_mu", torch.randn(1, self.hidden_dim))
        w_sigma = pyro.param("w_sigma", torch.randn(1, self.hidden_dim), constraint=dist.constraints.positive)
        b_mu = pyro.param("b_mu", torch.randn(self.hidden_dim))
        b_sigma = pyro.param("b_sigma", torch.randn(self.hidden_dim), constraint=dist.constraints.positive)

        # Approximate the distribution of weights with a Normal distribution
        w = pyro.sample("weights", dist.Normal(w_mu, w_sigma))
        b = pyro.sample("bias", dist.Normal(b_mu, b_sigma))
        
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        action_prob = F.softmax(self.layer3(h), dim=-1)
        return pyro.sample("action", dist.Categorical(action_prob))

import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class BayesianPolicyNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BayesianPolicyNN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def model(self, state, action):
        # Prior distribution for network weights
        fc1_w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
        fc1_b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
        
        fc2_w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
        fc2_b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))
        
        priors = {
            'fc1.weight': fc1_w_prior, 'fc1.bias': fc1_b_prior,
            'fc2.weight': fc2_w_prior, 'fc2.bias': fc2_b_prior
        }
        
        # Lift the priors to enable sampling from them
        lifted_module = pyro.random_module("module", self, priors)
        
        # Sample a neural network (which also samples w and b)
        lifted_nn = lifted_module()
        
        # Run input data through the neural network
        action_mean = lifted_nn(state)
        
        # Condition on the observed data
        pyro.sample("obs", dist.Normal(action_mean, 0.1), obs=action)
        
    def guide(self, state, action):
        # Variational parameters
        fc1_w_mu = torch.randn_like(self.fc1.weight)
        fc1_w_sigma = torch.randn_like(self.fc1.weight)
        fc1_w_mu_param = pyro.param("fc1_w_mu", fc1_w_mu)
        fc1_w_sigma_param = pyro.param("fc1_w_sigma", fc1_w_sigma, constraint=dist.constraints.positive)
        
        fc1_b_mu = torch.randn_like(self.fc1.bias)
        fc1_b_sigma = torch.randn_like(self.fc1.bias)
        fc1_b_mu_param = pyro.param("fc1_b_mu", fc1_b_mu)
        fc1_b_sigma_param = pyro.param("fc1_b_sigma", fc1_b_sigma, constraint=dist.constraints.positive)
        
        fc2_w_mu = torch.randn_like(self.fc2.weight)
        fc2_w_sigma = torch.randn_like(self.fc2.weight)
        fc2_w_mu_param = pyro.param("fc2_w_mu", fc2_w_mu)
        fc2_w_sigma_param = pyro.param("fc2_w_sigma", fc2_w_sigma, constraint=dist.constraints.positive)
        
        fc2_b_mu = torch.randn_like(self.fc2.bias)
        fc2_b_sigma = torch.randn_like(self.fc2.bias)
        fc2_b_mu_param = pyro.param("fc2_b_mu", fc2_b_mu)
        fc2_b_sigma_param = pyro.param("fc2_b_sigma", fc2_b_sigma, constraint=dist.constraints.positive)
        
        # Guide distributions
        fc1_w_dist = dist.Normal(loc=fc1_w_mu_param, scale=fc1_w_sigma_param)
        fc1_b_dist = dist.Normal(loc=fc1_b_mu_param, scale=fc1_b_sigma_param)
        
        fc2_w_dist = dist.Normal(loc=fc2_w_mu_param, scale=fc2_w_sigma_param)
        fc2_b_dist = dist.Normal(loc=fc2_b_mu_param, scale=fc2_b_sigma_param)
        
        dists = {
            'fc1.weight': fc1_w_dist, 'fc1.bias': fc1_b_dist,
            'fc2.weight': fc2_w_dist, 'fc2.bias': fc2_b_dist
        }
        
        # Overloading the parameters in the neural network with random samples
        lifted_module = pyro.random_module("module", self, dists)
        
        return lifted_module()


    def infer_action(bayesian_policy_nn, state, num_samples=100):
        sampled_actions = []
        for _ in range(num_samples):
            # This samples from the guide (posterior) and returns a 'new' model with those values
            sampled_nn = bayesian_policy_nn.guide(state, None)
            
            # Forward pass to get the action
            action_mean = sampled_nn(state)
            sampled_action = dist.Normal(action_mean, 0.1).sample()
            sampled_actions.append(sampled_action)
            
        # Averaging over all the samples to get the final action
        final_action = torch.stack(sampled_actions).mean(0)
        
        return final_action

# Example of how to use the infer_action function
##state = torch.FloatTensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)  # Replace this with a real state from your environment
##inferred_action = infer_action(bayesian_policy_nn, state)

### Initialize the Bayesian Neural Network
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

class BayesianValueNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(BayesianValueNN, self).__init__()
##        super(BayesianTRPOPolicy, self).__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear(hidden_dim, 1)
        
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
            sampled_action = dist.Normal(action_mean, 0.1).sample()
            sampled_actions.append(sampled_action)
            
        # Averaging over all the samples to get the final action
        final_action = torch.stack(sampled_actions).mean(0)
        
        return final_action

class BayesianTRPOValueN:
    def __init__(self, input_dim, hidden_dim, output_dim, environment, delta=0.01, gamma=0.99, lam=0.95):
        self.policy = BayesianTRPOPolicy(input_dim, hidden_dim, output_dim)
        self.value_net = BayesianRewardNetwork(input_dim) #, hidden_dim, output_dim, environment)  # Assuming a BayesianValueNetwork class exists
##        self.value_net = 
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


# Example usage:
state_dim = 4  # State dimension
action_dim = 2  # Action dimension
hidden_dim = 128  # Hidden layer dimension

policy = BayesianTRPOPolicyN(state_dim, hidden_dim, action_dim)

# Generate some fake data for state
state = torch.randn(5, state_dim)  # Batch of 5 states

# Model & guide should produce actions based on the state
action_mean, action_std = policy.model(state)
print("Model output:", action_mean, action_std)

action_mean, action_std = policy.guide(state)
print("Guide output:", action_mean, action_std)

# Implementing the rest (like natural gradients, Fisher-vector product, etc.) would follow.

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
            action = action_dist.sample()
##            next_state, reward, done, _ = env.step(action.item())
            next_state, reward, done = env.step(action.item())
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


    def run_episode(env, trpo_agent, hyperparams):
        learning_rate, entropy_weight, hidden_dim = hyperparams
        trpo_agent.update_hyperparams(learning_rate, entropy_weight, hidden_dim)
        
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action = trpo_agent.get_action(state_tensor)
            
            # Take action
            next_state, reward, done = env.step(action.cpu().numpy())
            
            # Update agent
            trpo_agent.update(state, action, reward, next_state, done)
            
            # Update the current state and episode reward
            state = next_state
            episode_reward += reward
        
        return episode_reward



class BayesianSharpeRatio:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, daily_return):
        self.alpha += daily_return if daily_return > 0.0 else 0.0
        self.beta += 1 - daily_return if daily_return <= 0.0 else 0.0

    def sample(self, num_samples):
        return beta.rvs(self.alpha, self.beta, size=num_samples)
    
    def calculate_kelly_bet(self):
        mean_return = self.alpha / (self.alpha + self.beta)
        variance_return = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2.0 * (self.alpha + self.beta + 1.0))
        
        p_win = self.alpha / (self.alpha + self.beta)
        K = (mean_return * p_win - (1.0 - p_win)) / mean_return

        return K

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
    
        reward = daily_return + expected_sharpe - self.balance * (1 - uncertainty)
        
        print('reward: ', reward)
        print('uncertainty: ', uncertainty)
        print('expected_sharpe: ', expected_sharpe)
##        print('sharpe_samples: ', sharpe_samples)
        print('optimal_bet_size: ', self.optimal_bet_size)
        
        return reward

##    def execute_trade(self, action):
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
        print(scaled_action)
        
        if scaled_action > 0.0:  
            max_buyable = min(self.balance / (price * (1.0 + self.transaction_fee)), self.balance)  
            actual_buy = scaled_action * max_buyable
            cost = actual_buy * price * (1.0 + self.transaction_fee)
            self.balance = max(0.0, self.balance - cost)
            self.btc_quantity += actual_buy
            print('actual_buy: ', actual_buy)



            
        elif scaled_action < 0.0:  
            max_sellable = self.btc_quantity
            actual_sell = -scaled_action * max_sellable
            revenue = actual_sell * price * (1.0 - self.transaction_fee)
            self.balance += revenue
            self.btc_quantity = max(0.0, self.btc_quantity - actual_sell)
            print('sell revenue: ', revenue)
            
    def get_state(self):
        print('self.current_step', self.current_step)        
        row = self.data.iloc[self.current_step]
        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
        return np.array([float(self.balance)] + float_values)
    
    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.start_step
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.initial_balance = 10000
        return self.get_state()

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
        
        self.current_step += 1
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        self.execute_trade(action * self.optimal_bet_size)  # Scale action by optimal bet size
        print('optimal_bet_size: ', self.optimal_bet_size)
        
        if self.current_step >= len(self.data):
            return self.reset(), 0, True

        self.execute_trade(action)
        
        reward = self.calculate_reward()
        print("Reward:", reward)  # Debugging line
        self.returns.append(reward.item() if torch.is_tensor(reward) else float(reward))
        
        done = self.current_step >= len(self.data) - 1
        
        return self.get_state(), reward, done

    def step1(self, actions):
        self.current_step += 1
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        self.execute_trade(action * self.optimal_bet_size)  # Scale action by optimal bet size
        print('optimal_bet_size: ', self.optimal_bet_size)
        
        if self.current_step >= len(self.data):
            return self.reset(), 0, True
        
        # Your existing code
        for idx, action in enumerate(actions):
            self.execute_trade(idx, action)

        # Update BayesianRewardCalculator
        new_return = (self.balance / self.initial_balance) - 1
        self.bayesian_reward_calculator.update(new_return)

        reward = self.bayesian_reward_calculator.get_reward()
        # Your existing code
        print("Reward:", reward)  # Debugging line
        self.returns.append(reward.item() if torch.is_tensor(reward) else float(reward))
        
        done = self.current_step >= len(self.data) - 1
        
        return self.get_state(), reward, done
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class DualPeaksActivation(nn.Module):
    def __init__(self):
        super(DualPeaksActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x) + torch.tanh(x)

class GradientBoostLayer(nn.Module):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super(GradientBoostLayer, self).__init__()
        self.boost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    def forward(self, x):
        x_numpy = x.detach().cpu().numpy()
        x_transformed = self.boost.predict(x_numpy)
        return torch.tensor(x_transformed, requires_grad=True).to(x.device)

class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, heads):
        super(EnhancedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout)
        self.attention = MultiheadAttention(hidden_dim, heads)
        self.dual_peaks = DualPeaksActivation()
        self.gradient_boost = GradientBoostLayer()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Applying the multi-head attention
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Dual peaks activation
        activated_out = self.dual_peaks(attention_out)
        
        # Gradient Boosting
        boosted_out = self.gradient_boost(activated_out)
        
        # Linear layer for output
        output = self.linear(boosted_out[-1])

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class BayesianLinear1(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear1, self).__init__()
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
        weight_sample = torch.normal(mean=self.weight_mu, std=self.softplus(self.weight_sigma))
        bias_sample = torch.normal(mean=self.bias_mu, std=self.softplus(self.bias_sigma))
        
        return F.linear(x, weight_sample, bias_sample)
class BayesianLinear(nn.Module):
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
        
        mixture = 0.5 * (beta1.sample() - 1) + 0.5 * (beta2.sample() + 1)
        
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

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam




# Assume BayesianLinear and BayesianLinear1 are defined somewhere
# ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.softmax(self.layer2(x), dim=1)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.layer1 = BayesianLinear1(input_dim, hidden_dim)
        self.layer2 = BayesianLinear1(hidden_dim, output_dim)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.layer2(x)
        return x


    


##    def run_episode(env, trpo_agent, hyperparams):
##        learning_rate, entropy_weight, hidden_dim = hyperparams
##        trpo_agent.update_hyperparams(learning_rate, entropy_weight, hidden_dim)
##        
##        state = env.reset()
##        done = False
##        episode_reward = 0
##        
##        while not done:
##            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
##            
##            # Get action
##            with torch.no_grad():
##                action = trpo_agent.get_action(state_tensor)
##            
##            # Take action
##            next_state, reward, done = env.step(action.cpu().numpy())
##            
##            # Update agent
##            trpo_agent.update(state, action, reward, next_state, done)
##            
##            # Update the current state and episode reward
##            state = next_state
##            episode_reward += reward
##        
##        return episode_reward
class TRPO:
    def __init__(self, policy_network, value_network, max_kl=0.01, damping=0.1):
        self.policy = policy_network
        self.value = value_network
        self.max_kl = max_kl
        self.damping = damping
        optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        value_optimizer = optim.Adam(self.value.parameters(), lr=0.001)
        policy_optimizer = optim.Adam(self.value.parameters(), lr=0.001)
        

    def sample_trajectory(self, env, state, T):
        states, actions, rewards = [], [], []
        for t in range(T):
            action_prob = self.policy(torch.FloatTensor(state))
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
##            next_state, reward, done, _ = env.step(action.item())
            next_state, reward, done = env.step(action.item())
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break
        return states, actions, rewards

    def run_episode(self, env):  # Added 'self' and removed hyperparameters for simplification
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_prob = self.policy(state_tensor)
                action_dist = Categorical(action_prob)
                action = action_dist.sample()

            next_state, reward, done = env.step(action.item())
            episode_reward += reward
            state = next_state
        
        return episode_reward

    def surrogate_loss(self, states, actions, advantages, old_probs=None):
        actions = torch.LongTensor(actions)
        
        new_probs = self.policy(states)
        action_masks = torch.zeros_like(new_probs)
        action_masks.scatter_(1, actions.view(-1, 1), 1)
        
        new_action_probs = torch.sum(new_probs * action_masks, dim=1)
        if old_probs is not None:
            old_action_probs = old_probs.squeeze()
        else:
            old_action_probs = new_action_probs.detach()  # Use the new probs as old if old_probs is None
        
        ratio = new_action_probs / old_action_probs
        surrogate_obj = ratio * advantages
        return surrogate_obj.mean()

##    
    def compute_advantages(self, states, rewards, gamma=0.99):
        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards))
        values = self.value(states).squeeze()
        deltas = rewards - values
        advantages = gamma * deltas
        return advantages


##    def surrogate_loss(self, states, actions, advantages, old_probs):
##        actions = torch.tensor(actions, dtype=torch.long)
##        new_probs = self.policy(states)
##        actions = actions.view(-1, 1)
##        selected_probs = torch.gather(new_probs, 1, actions)
##        selected_old_probs = torch.gather(old_probs, 1, actions).detach()
##
##        # To prevent division by zero
##        selected_old_probs += 1e-10
##
##        ratio = selected_probs / selected_old_probs
##        surrogate_obj = ratio * advantages.view(-1, 1)
##
##        return -torch.mean(surrogate_obj)


##    def update_policy(self, states, actions, advantages, rewards, gamma=0.99):
##        states = torch.FloatTensor(np.array(states))
##        rewards = torch.FloatTensor(np.array(rewards))
##        advantages = self.compute_advantages(states, rewards, gamma)
##
##        old_probs = self.policy(states).detach()
##        loss = self.surrogate_loss(states, actions, advantages, old_probs)
##
##        # Ensure only parameters with gradients are included
##        params_with_grad = [p for p in self.policy.parameters() if p.requires_grad]
##        grads = torch.autograd.grad(loss, params_with_grad, create_graph=True, allow_unused=True)
##        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
##
##        # Fisher-vector product function
##        def fvp(v):
##            kl = self.compute_kl(states)
##            grads = torch.autograd.grad(kl, params_with_grad, create_graph=True, allow_unused=True)
##            flat_grad_kl = torch.cat([grad.view(-1) if grad is not None else torch.zeros_like(p).view(-1) for grad, p in zip(grads, params_with_grad)])
##            kl_v = (flat_grad_kl * v).sum()
##            grads = torch.autograd.grad(kl_v, params_with_grad, allow_unused=True)
##            fisher_vector_product = torch.cat([grad.contiguous().view(-1) if grad is not None else torch.zeros_like(p).view(-1) for grad, p in zip(grads, params_with_grad)]).detach()
##            return fisher_vector_product + self.damping * v
##
##        step_direction = self.conjugate_gradient(fvp, flat_grads)
##
##        # Compute the natural gradient
##        natural_gradient = torch.sqrt(2 * self.max_kl / (torch.dot(step_direction, fvp(step_direction)) + 1e-8)) * step_direction
##
##        # Update the policy parameters
##        params = torch.cat([p.view(-1) for p in params_with_grad])
##        new_params = params + natural_gradient
##        self.vector_to_parameters(new_params, self.policy.parameters())
    
##    def conjugate_gradient(self, mat_vec_product, b, nsteps, residual_tol=1e-10):
##        p = b.clone()
##        r = b.clone()
##        x = torch.zeros_like(b)
##        r_norm = r.dot(r)
##        
##        for i in range(nsteps):
##            z = mat_vec_product(p)
##            alpha = r_norm / (p.dot(z) + 1e-8)
##            x += alpha * p
##            r -= alpha * z
##            r_norm_new = r.dot(r)
##            p = r + (r_norm_new / r_norm) * p
##            r_norm = r_norm_new
##            if r_norm < residual_tol:
##                break
                
##        return x

    def get_surrogate_loss(self, old_log_probs, states, actions, rewards, advantages):
        mean, log_std = self.policy(states)
        new_distribution = Normal(mean, log_std.exp())
        new_log_probs = new_distribution.log_prob(actions)
        
        ratio = (new_log_probs - old_log_probs).exp()
        surrogate_loss = - (ratio * advantages).mean()
        
        return surrogate_loss

    def update(self, states, actions, rewards, next_states, dones):
        # Compute advantage estimates based on the current value function
        with torch.no_grad():
            values = self.value(states)
            next_values = self.value(next_states)
        
        # Calculate the advantages
        advantages = rewards + (1 - dones) * 0.99 * next_values - values
        
        # Calculate old log probabilities of the actions
        mean, log_std = self.policy(states)
        old_distribution = Normal(mean, log_std.exp())
        old_log_probs = old_distribution.log_prob(actions)
        
        # Compute policy gradient and apply conjugate gradient to find search direction
        loss = self.get_surrogate_loss(old_log_probs.detach(), states, actions, rewards, advantages.detach())
        grads = autograd.grad(loss, self.policy.parameters())
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        
        def fisher_vector_product(x):
            return self.conjugate_gradient(lambda p: self.policy(p).dot(p), x, 10)
        
        # Compute step direction with Conjugate Gradient
        step_dir = self.conjugate_gradient(fisher_vector_product, -flat_grads, 10)

        # Compute optimal step size
        shs = 0.5 * step_dir.dot(fisher_vector_product(step_dir))
        lm = torch.sqrt(shs / 0.01)  # max kl
        full_step = step_dir / lm.item()
        
        # Update the policy network
        with torch.no_grad():
            flat_params = torch.cat([param.view(-1) for param in self.policy.parameters()])
            new_params = flat_params + full_step
            idx = 0
            for param in self.policy.parameters():
                param.copy_(new_params[idx: idx + param.numel()].view(param.size()))
                idx += param.numel()

        # Update the value network
        value_loss = ((rewards + 0.99 * next_values - self.value(states)) ** 2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
    def compute_kl(self, states):
        # Compute the KL divergence between old and new policies
        old_probs = self.policy(states).detach()
        new_probs = self.policy(states)
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(1)
        return kl.mean()
    
##    def compute_advantages(self, rewards, states, gamma=0.99):
##        state_values = self.value(torch.FloatTensor(states))
##        advantages = [0]
##        for i in reversed(range(len(rewards) - 1)):
##            delta = rewards[i] + gamma * state_values[i + 1] - state_values[i]
##            advantage = delta + gamma * advantages[-1]
##            advantages.insert(0, advantage)
##        return advantages

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
        

    def conjugate_gradient(self, fvp, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_old = torch.dot(r, r)
        for i in range(nsteps):
            Ap = fvp(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            beta = r_dot_new / (r_dot_old + 1e-10)
            p = r + beta * p
            r_dot_old = r_dot_new
            if r_dot_old < residual_tol:
                break
        return x

    def train(self, states, actions, old_probs, rewards, gamma, num_epochs=10):
        for _ in range(num_epochs):
            self.update_policy(states, actions, old_probs, rewards, gamma)

    def get_action(self, state):
        with torch.no_grad():
            action_prob = self.policy(state)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
        return action.item()

    def get_value(self, state):
        with torch.no_grad():
            value = self.value(state)
        return value.item()





class TRPO:
    def __init__(self, env, policy_network, value_network, optimizer, value_optimizer, max_kl=0.01, damping=0.1):
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.max_kl = max_kl
        self.damping = damping
        self.env=env
        
    def sample_trajectories(self, num_trajectories=10, max_timesteps=1000):
        trajectories = []
        
        for i in range(num_trajectories):
            observations = []
            actions = []
            rewards = []
            states = []
            
            obs = self.env.reset()
            for t in range(max_timesteps):
                action = self.policy_net.select_action(obs)
                
                next_obs, reward, done, _ = self.env.step(action)
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                states.append((obs, action, reward, next_obs, done))
                
                obs = next_obs
                
                if done:
                    break
                    
            trajectory = {'observations': np.array(observations), 
                          'actions': np.array(actions), 
                          'rewards': np.array(rewards), 
                          'states': states}
            
            trajectories.append(trajectory)
        
        return trajectories

    def sample_trajectory(env, state, max_length=1000):
        states, actions, rewards = [], [], []
##        state = env.reset()
        for _ in range(max_length):
            action = policy.select_action(state)
##            next_state, reward, done, _ = env.step(action)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        return states, actions, rewards

    def surrogate_loss(self, old_probs, new_probs, advantages):
        # Calculate the surrogate loss for TRPO
        ratio = new_probs / old_probs
        surrogate_obj = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.max_kl, 1 + self.max_kl) * advantages)
        return -surrogate_obj.mean()

    def compute_advantages(self, states, rewards, gamma):
        # Calculate advantages using the value network
        values = self.value_network(states).squeeze()
        deltas = rewards - values
        advantages = []
        advantage = 0
        for delta in deltas[::-1]:
            advantage = delta + gamma * advantage
            advantages.append(advantage)
        advantages = torch.tensor(advantages[::-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update_policy(self, states, actions, old_probs, rewards, gamma):
        # Calculate advantages
        advantages = self.compute_advantages(states, rewards, gamma)

        # Calculate the gradient of the surrogate loss
        new_probs = self.policy_network(states).gather(1, actions.unsqueeze(1))
        loss = self.surrogate_loss(old_probs, new_probs, advantages)
        grads = torch.autograd.grad(loss, self.policy_network.parameters())

        # Compute the Fisher information matrix vector product (Fisher-vector product)
        grads = torch.cat([grad.view(-1) for grad in grads])
        fisher_vector_product = self.compute_fvp(states, advantages, grads)

        # Compute the step direction using conjugate gradient
        step_direction = self.conjugate_gradient(states, fisher_vector_product)

        # Compute the natural gradient
        natural_gradient = torch.sqrt(2 * self.max_kl / (torch.dot(step_direction, fisher_vector_product) + 1e-8)) * step_direction

        # Compute the new policy parameters
        params = torch.cat([param.view(-1) for param in self.policy_network.parameters()])
        new_params = params + natural_gradient

        # Update the policy network parameters
        self.update_policy_parameters(new_params)

    def compute_fvp(self, states, advantages, vector):
        # Compute the Fisher-vector product
        kl = self.compute_kl(states)
        grads = torch.autograd.grad(kl, self.policy_network.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * vector).sum()
        grads = torch.autograd.grad(kl_v, self.policy_network.parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
        return fisher_vector_product + self.damping * vector

    def conjugate_gradient(self, states, fisher_vector_product):
        # Solve the linear system using conjugate gradient
        x = torch.zeros_like(fisher_vector_product)
        r = fisher_vector_product.clone()
        p = r.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(10):  # Adjust the number of iterations as needed
            Ap = self.compute_fvp(states, advantages, p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            beta = r_dot_new / (r_dot_old + 1e-8)
            p = r + beta * p
            r_dot_old = r_dot_new
        return x
    def compute_kl(self, states):
        # Compute the KL divergence between old and new policies
        old_probs = self.old_policy(states)
        new_probs = self.policy_network(states)
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(1)
        return kl.mean()

    def old_policy(self, states):
        # Compute the probability of old policy
        with torch.no_grad():
            return self.policy_network(states)

    def update_policy_parameters(self, new_params):
        # Update policy network parameters
        current_pos = 0
        for param in self.policy_network.parameters():
            param_size = param.numel()
            new_param = new_params[current_pos: current_pos + param_size]
            param.data = new_param.view(param.size())
            current_pos += param_size

    def compute_inverse_fim(self, states, advantages):
        # Compute the Inverse Fisher Information Matrix
        kl = self.compute_kl(states)
        grads = torch.autograd.grad(kl, self.policy_network.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        fisher_information_matrix = torch.autograd.grad(flat_grad_kl, self.policy_network.parameters(), create_graph=True)
        flat_fisher_information_matrix = torch.cat([grad.view(-1) for grad in fisher_information_matrix])
        inverse_fim = torch.linalg.pinv(flat_fisher_information_matrix)  # Pseudo-inverse
        return inverse_fim



    
    def compute_value(self, states):
        # Calculate state values using the value network
        return self.value_network(states)

    def update_value_network(self, states, targets):
        # Update the value network using mean squared error loss
        predicted_values = self.value_network(states)
        value_loss = nn.MSELoss()(predicted_values, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def optimize_policy(self, states, actions, old_probs, rewards, gamma):
        # Calculate advantages
        advantages = self.compute_advantages(states, rewards, gamma)

        # Calculate the gradient of the surrogate loss
        new_probs = self.policy_network(states).gather(1, actions.unsqueeze(1))
        policy_loss = self.surrogate_loss(old_probs, new_probs, advantages)

        # Compute the KL divergence
        kl = self.compute_kl(states)

        # Compute the gradient of the KL divergence
        grads = torch.autograd.grad(kl, self.policy_network.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        # Compute the Fisher Information Matrix
        fisher_information_matrix = torch.autograd.grad(flat_grad_kl, self.policy_network.parameters(), create_graph=True)

        # Compute the Inverse Fisher Information Matrix
        flat_fisher_information_matrix = torch.cat([grad.view(-1) for grad in fisher_information_matrix])
        inverse_fim = torch.linalg.pinv(flat_fisher_information_matrix)

        # Compute the natural gradient
        natural_gradient = torch.matmul(inverse_fim, flat_grad_kl)

        # Compute the step direction using the natural gradient
        step_direction = torch.sign(natural_gradient) * torch.sqrt(2 * self.max_kl)

        # Compute the new policy parameters
        params = torch.cat([param.view(-1) for param in self.policy_network.parameters()])
        new_params = params + step_direction

        # Update the policy network parameters
        self.update_policy_parameters(new_params)

        # Update the value network
        self.update_value_network(states, rewards)
    

from itertools import product
from tqdm import tqdm
def main():
    # Hyperparameter space
    hyperparam_space = {
        'learning_rate': [1e-3, 3e-4, 1e-4],
        'entropy_weight': [0.1, 0.2, 0.3],
        'hidden_dim': [128, 256, 512],
        'transaction_fee': [0.001, 0.01],
    }
    
    # Generate all combinations of hyperparameters
    hyperparam_combinations = list(product(*hyperparam_space.values()))
    
    best_reward = -float('inf')
    best_hyperparams = None
    
    # Loop over all hyperparameter combinations
    for hyperparams in tqdm(hyperparam_combinations):
        
        learning_rate, entropy_weight, hidden_dim, transaction_fee = hyperparams
        ### Parameters to tune ###
        learning_rate = 3e-4
        entropy_weight = 0.2
        hidden_dim = 1024

        initial_balance =1000
        transaction_fee=0.001
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

        state = env.reset()
        done = False
        episode_reward = 0
        actions = []
        
        policy_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        value_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
        value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
        trpo = TRPO(env, policy_network, value_network, optimizer, value_optimizer)


##        trpo = SACAgent(state_dim, action_dim, hidden_dim, learning_rate, entropy_weight)
        
        logging.info(f'state dim = {state_dim}')


        logging.info(f'state dim = {state_dim}')







        ### Example usage
        input_dim = len(env.get_state())
        action_dim = 1
        hidden_dim = 128
        hidden2_dim = 64
        output_dim = 1

##        # Initialize Environment, Policy, and Value network
##        # env = YourEnvironmentHere()
##        policy_net = BayesianLinear1(state_dim, hidden_dim)
##        value_net = BayesianLinear(state_dim, 1)
####
####        # Initialize TRPO
##        trpo = TRPO(policy_net, value_net)

##        policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
##        value_network = ValueNetwork(input_dim, hidden_dim, 1)
        
##        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
##        value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##        policy_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##        trpo = TRPO(policy_network, value_network)


##        # Training loop
##        for episode in range(100):
##            state = env.reset()
##            states, actions, rewards = trpo.sample_trajectory(env, state, T=1000)
##            print(states)
##            advantages = trpo.compute_advantages(rewards, states)
##            print(advantages)
##
##            trpo.update_policy(states, actions, advantages)




        # Training loop
        n_episodes = num_episodes
        with tqdm(total=n_episodes) as pbar:
            for episode in tqdm(range(n_episodes)):
                pbar.set_postfix({"Episode": f"{episode+1}/{n_episodes}"}, refresh=True)
                
                state = env.reset()
                done = False
                episode_reward = 0
                actions = []
                while not done:

                    states, actions, rewards = trpo.sample_trajectory(env, state)#, T=1006)
                    print(states)
                    advantages = trpo.compute_advantages(rewards, states, gamma=0.9)
                    print(advantages)

##                    trpo.update_policy(states, actions, advantages)

                    trpo.update_policy(states, actions, advantages, rewards)#, gamma=0.99)

# Main function
if __name__ == '__main__':
    main()




    
