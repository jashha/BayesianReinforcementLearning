import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyro.infer import SVI, Trace_ELBO
import torch.optim as optim
import pandas as pd
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
from torch.nn import functional as F
import torch
import torch.nn as nn
from scipy.stats import beta
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import torch
import pyro
import pyro.distributions as dist
from pyro.#infer import SVI, Trace_ELBO
from pyro.optim import Adam




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)


'''
Define env
'''





class BayesianLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn='leakyrelu'):
        super(BayesianLayer1, self).__init__()
        
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))
        
        if activation_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        weight_sample = Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias_sample = Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()
        output = F.linear(x, weight_sample, bias_sample)
        output = self.activation(output)
        return output

class BayesianLayer2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLayer2, self).__init__()
        
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
class BayesianLayer1(BayesianNN2):
    #outputs a layer
    def __init__(self, input_dim, output_dim, activation_fn='leakyrelu'):
        super(BayesianLayer, self).__init__()
        
        # Initialize means and standard deviations for weights
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-5, -4))
        
        # Initialize means and standard deviations for biases
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))
        
        # Activation function
        if activation_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation_fn == 'relu':
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

class BayesianLayer2(BayesianNN2):
    # outputs mean std
    def __init__(self, input_dim, output_dim):
        super(BayesianLayer, self).__init__()
        
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
### Assuming your CryptoEnvironment looks somewhat like this
####class CryptoEnvironment:
####    def __init__(self, config):
####
####        
####    def step(self, action):
####        # Your logic to transition from current_state to next_state
####        # ...
####        
####        reward = self.reward_calculator.calculate_reward(self.current_state, self.next_state)
####        return next_state, reward, done, info
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
    
    def render(self):
        print(f"Current Step: {self.current_step} out of {len(self.data)}")
        print(f"Current Balance: {self.balance}")
        print(f"BTC quantity Owned:  {self.btc_quantity}")
##        print(f"returns:  {self.returns}")
        
    def _calculate_reward(self):
        daily_return = (self.balance / self.initial_balance) - 1.0
        sharpe_ratio = 0.0 if len(self.returns) < 2 else np.mean(self.returns) / np.std(self.returns)
        print("Sharpe Ratio:", sharpe_ratio)  # Debugging line
        return daily_return + sharpe_ratio




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
##        self.hidden1 = BayesianLayer1(input_dim, hidden1_dim)
##        self.hidden2 = BayesianLayer1(hidden1_dim, hidden2_dim)
##        self.output = BayesianLayer1(hidden2_dim, output_dim)
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
##        x = F.relu(F.linear(x, weight=torch.normal(mean=self.hidden1_mean, std=self.hidden1_std), bias=self.hidden1.bias))
##        x = F.relu(F.linear(x, weight=torch.normal(mean=self.hidden2_mean, std=self.hidden2_std), bias=self.hidden2.bias))
##        x = F.linear(x, weight=torch.normal(mean=self.output_mean, std=self.output_std), bias=self.output.bias)
##
##        x = BayesianLayer1(mean=self.output_mean, std=self.output_std)
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


class BayesianActor(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim=2):
        super(BayesianActor, self).__init__()
        
        # Initialize mean and std as learnable parameters
        self.hidden1_mean = nn.Parameter(torch.Tensor(hidden1_dim, input_dim))
        self.hidden1_std = nn.Parameter(torch.Tensor(hidden1_dim, input_dim))
        
        self.hidden2_mean = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
        self.hidden2_std = nn.Parameter(torch.Tensor(hidden2_dim, hidden1_dim))
        
        self.output_mean = nn.Parameter(torch.Tensor(output_dim, hidden2_dim))
        self.output_std = nn.Parameter(torch.Tensor(output_dim, hidden2_dim))

        nn.init.xavier_uniform_(self.hidden1_mean)
        nn.init.constant_(self.hidden1_std, 0.05)

        nn.init.xavier_uniform_(self.hidden2_mean)
        nn.init.constant_(self.hidden2_std, 0.05)

        nn.init.xavier_uniform_(self.output_mean)
        nn.init.constant_(self.output_std, 0.05)

        self.to(device)

    def forward(self, x):
        x = x.to(device)
        
        # Sample weights from normal distribution
        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std)
        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std)
        output_weight = torch.normal(mean=self.output_mean, std=self.output_std)

        # Perform forward pass
        x = F.relu(F.linear(x, weight=hidden1_weight))
        x = F.relu(F.linear(x, weight=hidden2_weight))
        x = F.linear(x, weight=output_weight)
        
        output_mean = x[:, 0]
        output_std = F.softplus(x[:, 1])
        print(f"output_std : {output_std}")

        output_std = torch.clamp(output_std, min=1e-3)
        print(f"output_std : {output_std}")
        print(f"output_mean: {output_mean}")        
        return output_mean, output_std

####### Test the BayesianActor class
######if __name__ == '__main__':


input_dim = 5
hidden1_dim = 10
hidden2_dim = 10
output_dim = 2

model = BayesianActor(input_dim, hidden1_dim, hidden2_dim, output_dim)
x = torch.randn(3, input_dim)

output_mean, output_std = model(x)
print(f'output_mean: {output_mean}')
print(f'output_std: {output_std}')

'''
Define Critic
'''
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
        
##        # Debugging: print shapes
        print("State shape:", state.shape)
        print("Action shape:", action.shape)
##
##        # Make sure action is a 2D tensor
##        if len(action.shape) == 1:
##            action = action.unsqueeze(-1)
##
##        # Concatenate state and action
##        x = torch.cat([state, action], dim=-1)
        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)

        # Sample weights (not modifying the class attributes)
##        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std).to(device)
##        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std).to(device)
##        output_weight = torch.normal(mean=self.output_mean, std=self.output_std).to(device)
##
##        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
##        x = F.relu(x)
##
##        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
##        x = F.relu(x)
##
##        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)
##
##        return q_value

        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std))

        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std).to(device)
        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std).to(device)
        output_weight = torch.normal(mean=self.output_mean, std=self.output_std).to(device)
        
##        x = F.relu(self.hidden1(x))
##        x = F.relu(self.hidden2(x))
##        q_value = self.output(x)
##        q_value = self.output(x)

        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
        x = F.relu(x)

        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
        x = F.relu(x)

        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)

        return q_value
'''
SAC agent
'''
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=3e-4, alpha=0.2, lr_alpha=1e-4):
        self.actor = BayesianActor(state_dim, hidden_dim, hidden_dim, 2).to(device)  # Mean and std
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
        q_value1 = self.critic1(states, actions)
        q_value2 = self.critic2(states, actions)
##    def update(self, states, actions, rewards, next_states, dones):
##        # Compute the Q-values
##        q_value1 = self.critic1(states, actions)
##        q_value2 = self.critic2(states, actions)

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
##        actor_loss.backward()
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
##        x = torch.cat([state, action.unsqueeze(-1)], dim=-1)
##        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
##        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
##        self.output.weight = nn.Parameter(torch.normal(mean=self.output_mean, std=self.output_std))
##
##        hidden1_weight = torch.normal(mean=self.hidden1_mean, std=self.hidden1_std).to(device)
##        hidden2_weight = torch.normal(mean=self.hidden2_mean, std=self.hidden2_std).to(device)
##        output_weight = torch.normal(mean=self.output_mean, std=self.output_std).to(device)
##        x = F.linear(input=x, weight=hidden1_weight, bias=self.hidden1.bias)
##        x = F.relu(x)
##
##        x = F.linear(input=x, weight=hidden2_weight, bias=self.hidden2.bias)
##        x = F.relu(x)
##
##        q_value = F.linear(input=x, weight=output_weight, bias=self.output.bias)
##
##        return q_value



import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
##        self.hidden1.weight = nn.Parameter(torch.normal(mean=self.hidden1_mean, std=self.hidden1_std))
##        self.hidden2.weight = nn.Parameter(torch.normal(mean=self.hidden2_mean, std=self.hidden2_std))
##        self.action_mean.weight = nn.Parameter(torch.normal(mean=self.action_mean_mean, std=self.action_mean_std))
##        self.action_std.weight = nn.Parameter(torch.normal(mean=self.action_std_mean, std=self.action_std_std))
##        
##        x = F.linear(input=state, weight=self.hidden1.weight, bias=self.hidden1.bias)
##        x = F.relu(x)
##
##        x = F.linear(input=x, weight=self.hidden2.weight, bias=self.hidden2.bias)
##        x = F.relu(x)
##
##        mean = self.action_mean(x)
##        std = F.softplus(self.action_std(x))  # Ensure std is positive
##
##        return mean, std



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
            action = action_dist.sample()
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
                state_value = state_value_dist.sample()

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
    def __init__(self, input_dim, hidden_dim=1):
        super(BayesianSharpeRatioNN, self).__init__()
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
        sac_agent = SACAgent(state_dim, action_dim, hidden_dim, learning_rate, entropy_weight)
        
        logging.info(f'state dim = {state_dim}')
        # Initialize environment and agent
        # Training loop
        n_episodes = num_episodes
        for episode in tqdm(range(n_episodes)):
            pbar.set_postfix({"Episode": f"{episode+1}/{n_episodes}"}, refresh=True)
            
            state = env.reset()
            done = False
            episode_reward = 0
            actions = []
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
##                        action_mean, action_std = sac_agent.actor(state_tensor)
##                        action_distribution = Normal(action_mean, action_std)
##                        sampled_action = action_distribution.sample()
##                        tanh_action = torch.tanh(sampled_action)
                    
                    action_mean, action_std = sac_agent.actor(state_tensor)
                    action_distribution = torch.distributions.Normal(action_mean, action_std)
                    print('action is action_distribution :', action_distribution)
                    sampled_action = action_distribution.sample()
                    print('action is after sampled_action :', sampled_action)

                    action = torch.tanh(action_distribution.sample()) 
                    print('action is mains get action :', action)
                    
##                    print(actions)
                actions.append(action) # Print to see!!!
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
        
            action_mean, action_std = sac_agent.actor(state_tensor)
            action_std = torch.clamp(action_std, min=1e-3)
            action_distribution = torch.distributions.Normal(action_mean, action_std)
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





