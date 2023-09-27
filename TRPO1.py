import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn
##from sqlalchemy import create_engine
##from sqlalchemy.orm import sessionmaker
##from sqlalchemy.ext.declarative import declarative_base
import os
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
import pyro
import pyro.distributions as dist
from torch import nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)


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


import torch
from torch.distributions import Categorical, Normal, Beta
##
##class GMM_Policy:
##    def __init__(self, mu1, std1, mu2, std2, alpha, beta):
##        self.gaussian1 = Normal(mu1, std1)
##        self.gaussian2 = Normal(mu2, std2)
##        self.alpha = alpha
##        self.beta = beta
##
##    def sample_trajectory(self, env, state, T):
##        states, actions, rewards = [], [], []
##        for t in range(T):
##            # Sample from Gaussian mixture model
##            if torch.rand(1) < 0.5:
##                sample = self.gaussian1.sample()
##            else:
##                sample = self.gaussian2.sample()
##
##            # Map to Beta distribution
##            action_prob = Beta(self.alpha, self.beta).cdf(sample)
##            
##            # Convert to categorical (optional)
##            action_dist = Categorical(torch.tensor([1-action_prob, action_prob]))
##            action = action_dist.sample()
##            
##            next_state, reward, done = env.step(action.item())
##            states.append(state)
##            actions.append(action)
##            rewards.append(reward)
##            state = next_state
##            if done:
##                break
##        return states, actions, rewards
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##import torch.distributions as D
####
##class GMM_Policy(nn.Module):
##    def __init__(self, input_dim, output_dim):
##        super(GMM_Policy, self).__init__()
##        self.fc1 = nn.Linear(input_dim, 128)
##        self.fc2 = nn.Linear(128, 64)
##        
##        self.mu_head = nn.Linear(64, output_dim * 2)  # Two peaks
##        self.sigma_head = nn.Linear(64, output_dim * 2)  # Two peaks
##        self.pi_head = nn.Linear(64, 2)  # Mixture weights for two peaks
##
##    def forward(self, x):
##        x = F.relu(self.fc1(x))
##        x = F.relu(self.fc2(x))
##
##        mu = self.mu_head(x)
##        sigma = F.softplus(self.sigma_head(x))
##        pi = F.softmax(self.pi_head(x), dim=1)
##
##        return mu, sigma, pi








class TRPO:
    def __init__(self, policy_network, value_network, optimizer, value_optimizer, max_kl=0.01, damping=0.1):
        self.policy = policy_network
##        self.policy = GMM_Policy()
        self.value = value_network
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.max_kl = max_kl
        self.damping = damping
##        self.svi = SVI(self.value.model, self.value.guide, self.value_optimizer, loss=Trace_ELBO())
        


    def surrogate_loss(self, old_probs, new_probs, advantages):
        # Calculate the surrogate loss for TRPO
        ratio = new_probs / old_probs
        surrogate_obj = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.max_kl, 1 + self.max_kl) * advantages)
        return -surrogate_obj.mean()

##    def compute_advantages(self, states, rewards, gamma):
##        # Calculate advantages using the value network
##        states_tensor = torch.Tensor(states)  # Convert list to tensor
##        values = self.value(states_tensor).squeeze()
##        values = self.value(states).squeeze()
##        states_tensor = states_tensor.to(device)  # 'device' should be where your model is, for example, "cuda:0"
##        values = self.value(states_tensor).squeeze()
##    def compute_advantages(self, rewards, states, gamma=0.99):
##        rewards = torch.tensor(rewards, dtype=torch.float32)  # Convert rewards to tensor
##        values = self.value(states).squeeze()
##        print(f"Rewards shape: {rewards.shape}, Values shape: {values.shape}")  # Debug shapes
##
##        deltas = rewards - values
##        print("Length of deltas: ", len(deltas))
##        print(f"Deltas shape: {deltas.shape}, Deltas: {deltas}")  # Debug deltas
##
##        if len(deltas.shape) == 0:  # Check if deltas is a scalar
##            raise ValueError("Deltas is a scalar. Expected a tensor.")
##        if deltas.shape[0] == 0:  # Check if deltas is empty
##            raise ValueError("Deltas tensor is empty.")
##
##        advantages = []
##        advantage = 0
##
##        deltas_list = deltas.tolist()
##        for delta in reversed(deltas_list):
##            advantage = delta + gamma * advantage
##            advantages.append(advantage)
##
##        advantages = torch.tensor(advantages[::-1])
##        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
##
##        return advantages
    
##    def compute_advantages(self, rewards, states, gamma=0.99):
##        state_values = self.value(torch.FloatTensor(states))
##        advantages = [0]
##        for i in reversed(range(len(rewards) - 1)):
##            delta = rewards[i] + gamma * state_values[i + 1] - state_values[i]
##            advantage = delta + gamma * advantages[-1]
##            advantages.insert(0, advantage)
##        return advantages


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


##    def sample_trajectory(self, env, state, T):
##        states, actions, rewards = [], [], []
##        for t in range(T):
##            action_prob = self.policy(torch.FloatTensor(state))
##            action_dist = Categorical(action_prob)
##            action = action_dist.sample()
####            next_state, reward, done, _ = env.step(action.item())
##            next_state, reward, done = env.step(action.item())
##            states.append(state)
##            actions.append(action)
##            rewards.append(reward)
##            state = next_state
##            if done:
##                break
##        return states, actions, rewards


    def sample_trajectory(self, env, initial_state, T):
        """
        Sample a trajectory from the environment using the current policy.
        
        Parameters:
        - env: The environment object
        - initial_state: Initial state to start sampling from
        - T: Maximum number of steps to sample
        
        Returns:
        - states: List of states visited
        - actions: List of actions taken
        - rewards: List of rewards received
        """
        states, actions, rewards = [], [], []
        state = initial_state
        
        for t in range(T):
            # Forward pass through the policy network
            action_prob = self.policy(torch.FloatTensor(state))
            
            # Sample action from the categorical distribution
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            
            # Take action in the environment
            next_state, reward, done = env.step(action.item())
            
            # Store state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Update current state
            state = next_state
            
            # If episode terminates, break
            if done:
                break
                
        return states, actions, rewards

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

import torch.optim as optim

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Define your policy and value network classes here.
# For example:
# class PolicyNetwork(nn.Module):
#     ...

# class ValueNetwork(nn.Module):
#     ...
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

# Custom Dual Peaks activation function
class DualPeaksActivation(nn.Module):
    def __init__(self):
        super(DualPeaksActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x) + torch.tanh(x)

# Simple neural network model
class GMM_DualPeaks_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_components=3):
        super(GMM_DualPeaks_Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dual_peaks = DualPeaksActivation()
        self.gmm = GaussianMixture(n_components=n_components)
        self.fitted = False

##    def fit_gmm(self, x):
##        x_numpy = x.detach().cpu().numpy()
####        self.gmm.fit(x_numpy)
##        # If x_numpy is a single feature array
####        self.gmm.fit(x_numpy.reshape(-1, 1))
##
##        # OR if x_numpy is a single sample array
##        self.gmm.fit(x_numpy.reshape(1, -1))
##        self.fitted = True

    def fit_gmm(self, x):
        x_numpy = x.detach().cpu().numpy()

        if x_numpy.shape[0] < 2:  # Check the number of samples
            print("Insufficient samples for GMM. Need at least 2 but got", x_numpy.shape[0])
            return  # Return without fitting
        
        # Now the shape should be compatible, but you might still want to reshape it to make sure
        if len(x_numpy.shape) == 1:
            x_numpy = x_numpy.reshape(1, -1)
            
        self.gmm.fit(x_numpy)
        self.fitted = True

    def apply_gmm(self, x):
        if not self.fitted:
            raise RuntimeError("GMM not fitted yet.")
        x_numpy = x.detach().cpu().numpy()
##        x_transformed = self.gmm.predict_proba(x_numpy)
        # In the `apply_gmm` method:
        x_transformed = self.gmm.predict_proba(x_numpy.reshape(1, -1))
        return torch.tensor(x_transformed, dtype=torch.float).to(x.device)

    def forward(self, x):
        if not self.fitted:
            self.fit_gmm(x)
        x = self.apply_gmm(x)
        x = self.layer1(x)
        x = self.dual_peaks(x)
        x = self.layer2(x)
        return x

### Create the model
##input_dim = 10
##hidden_dim = 20
##output_dim = 5
##model = GMM_DualPeaks_Network(input_dim, hidden_dim, output_dim)
##
### Create some random input
##x = torch.rand((50, input_dim))
##
##### Forward pass
####output = model(x)
####print(output)
##class TRPO:
##    def __init__(self, policy_network, value_network, max_kl=0.01, damping=0.1):
##        self.policy = policy_network
##        self.value = value_network
##        self.optimizer = optim.Adam(self.policy.parameters())
##        self.value_optimizer = optim.Adam(self.value.parameters())
##        self.max_kl = max_kl
##        self.damping = damping
##
##    def surrogate_loss(self, old_probs, new_probs, advantages):
##        ratio = new_probs / old_probs
##        surr_obj = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.max_kl, 1 + self.max_kl) * advantages)
##        return -surr_obj.mean()
##
##    def compute_advantages(self, rewards, states, gamma=0.99):
##        rewards = torch.tensor(rewards, dtype=torch.float32)
##        values = self.value(states).squeeze()
##
##        deltas = rewards - values
##        advantages = []
##        advantage = 0
##        for delta in reversed(deltas.tolist()):
##            advantage = delta + gamma * advantage
##            advantages.append(advantage)
##
##        advantages = torch.tensor(advantages[::-1])
##        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
##        return advantages
##
##    def update(self, states, actions, rewards):
##        # Convert to tensor
##        states = torch.tensor(states, dtype=torch.float32)
##        actions = torch.tensor(actions, dtype=torch.int64)
##        
##        # Compute advantages
##        advantages = self.compute_advantages(rewards, states)
##
##        # Compute old action probabilities
##        old_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
##
##        # Optimize policy using TRPO updates
##        self.optimizer.zero_grad()
##        new_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
##        loss = self.surrogate_loss(old_probs.detach(), new_probs, advantages.detach())
##        loss.backward()
##        self.optimizer.step()
##
##        # Update the value network
##        self.value_optimizer.zero_grad()
##        targets = rewards + 0.99 * self.value(states[1:]).squeeze()
##        value_loss = F.mse_loss(self.value(states[:-1]).squeeze(), targets.detach())
##        value_loss.backward()
##        self.value_optimizer.step()



# Assume we have some training loop and we collect states, actions, and rewards
# states, actions, rewards = collect_data()
# trpo_agent.update(states, actions, rewards)
##class TRPO:
##    def __init__(self, input_dim, output_dim, value_network, optimizer, value_optimizer, max_kl=0.01, damping=0.1):
##        self.policy = GMM_Policy(input_dim, output_dim)
##        self.value = value_network
##        self.optimizer = optimizer
##        self.value_optimizer = value_optimizer
##        self.max_kl = max_kl
##        self.damping = damping
##
##    def get_action(self, state):
##        state = torch.Tensor(state).unsqueeze(0)  # Add batch dimension
##        mu, sigma, pi = self.policy(state)
##        
##        # Create GMM
##        components = D.Normal(mu, sigma)
##        mixture = D.Categorical(pi)
##        
##        # Sample action
##        component_idx = mixture.sample().item()
##        action = components[component_idx].sample().item()
##        
##        return action


# Assume GMM_Policy and TRPO initial definition are above

##    def surrogate_loss(self, old_probs, new_probs, advantages):
##        ratio = new_probs / (old_probs + 1e-8)
##        clipped_ratio = torch.clamp(ratio, 1.0 - self.max_kl, 1.0 + self.max_kl)
##        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
##        return -torch.mean(surrogate)
##
##    def compute_advantages(self, states, rewards, gamma=0.99):
##        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
##        states = torch.tensor(states, dtype=torch.float32).to(device)
##        values = self.value(states).squeeze(-1)
##
##        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
##        advantages = torch.zeros_like(rewards).to(device)
##
##        advantage = 0.0
##        for t in reversed(range(len(rewards) - 1)):
##            advantage = deltas[t] + gamma * advantage
##            advantages[t] = advantage
##
##        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
##        return advantages
##
##    def update(self, states, actions, rewards):
##        states = torch.tensor(states, dtype=torch.float32).to(device)
##        actions = torch.tensor(actions, dtype=torch.float32).to(device)
##        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
##
##        # Compute advantages
##        advantages = self.compute_advantages(states, rewards)
##
##        # Compute old action probabilities
##        mu, sigma, pi = self.policy(states)
##        old_probs = D.Normal(mu, sigma).log_prob(actions).exp() * pi
##        old_probs = old_probs.sum(dim=1)
##
##        def get_loss():
##            mu, sigma, pi = self.policy(states)
##            new_probs = D.Normal(mu, sigma).log_prob(actions).exp() * pi
##            new_probs = new_probs.sum(dim=1)
##            return self.surrogate_loss(old_probs.detach(), new_probs, advantages.detach())

        # Perform TRPO update here (omitted for brevity)
        # You may use conjugate gradients, line search, etc.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

##class TRPO:
##    def __init__(self, 
##                 policy_network: nn.Module, 
##                 value_network: nn.Module, 
##                 optimizer: optim.Optimizer, 
##                 value_optimizer: optim.Optimizer, 
##                 max_kl: float = 0.01, 
##                 damping: float = 0.1):
##        """
##        Initialize TRPO agent.
##        """
##        self.policy = policy_network
##        self.value = value_network
##        self.optimizer = optimizer
##        self.value_optimizer = value_optimizer
##        self.max_kl = max_kl
##        self.damping = damping
##
##    def surrogate_loss(self, old_probs: torch.Tensor, new_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
##        """
##        Compute the TRPO surrogate loss.
##        """
##        ratio = new_probs / old_probs
##        surrogate_obj = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.max_kl, 1 + self.max_kl) * advantages)
##        return -surrogate_obj.mean()
##
##    def compute_advantages(self, rewards: torch.Tensor, states: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
##        """
##        Compute advantages using the value network.
##        """
##        rewards = torch.tensor(rewards, dtype=torch.float32).to(states.device)  
##        values = self.value(states).squeeze()
##        
##        deltas = rewards - values
##        if deltas.dim() == 0 or deltas.size(0) == 0:
##            raise ValueError("Deltas tensor is either scalar or empty.")
##        
##        advantages = torch.zeros_like(rewards)
##        advantage = 0.0
##        for t in reversed(range(len(rewards))):
##            advantage = rewards[t] + gamma * advantage - values[t]
##            advantages[t] = advantage
##
##        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Add other methods like optimize_policy, compute_value, update_value_network, etc.


##    def sample_trajectory(self, env, initial_state, T):
##        """
##        Sample a trajectory from the environment using the current policy.
##        
##        Parameters:
##        - env: The environment object
##        - initial_state: Initial state to start sampling from
##        - T: Maximum number of steps to sample
##        
##        Returns:
##        - states: List of states visited
##        - actions: List of actions taken
##        - rewards: List of rewards received
##        """
##        states, actions, rewards = [], [], []
##        state = initial_state
##        
##        for t in range(T):
##            # Forward pass through the policy network
##            action_prob = self.policy(torch.FloatTensor(state))
##            
##            # Sample action from the categorical distribution
##            action_dist = Categorical(action_prob)
##            action = action_dist.sample()
##            
##            # Take action in the environment
##            next_state, reward, done = env.step(action.item())
##            
##            # Store state, action, and reward
##            states.append(state)
##            actions.append(action)
##            rewards.append(reward)
##            
##            # Update current state
##            state = next_state
##            
##            # If episode terminates, break
##            if done:
##                break
##                
##        return states, actions, rewards
##
##    def compute_kl(self, old_probs: torch.Tensor, new_probs: torch.Tensor) -> torch.Tensor:
##        """
##        Compute the KL divergence between the old and new action probabilities.
##        """
##        return (old_probs * torch.log(old_probs / new_probs)).sum()
##
##    def update_policy_parameters(self, new_params: torch.Tensor):
##        """
##        Update the policy network parameters.
##        """
##        idx = 0
##        for param in self.policy.parameters():
##            param_length = param.nelement()
##            param.data = new_params[idx:idx + param_length].view(param.size())
##            idx += param_length
##
##    def optimize_policy(self, states: torch.Tensor, actions: torch.Tensor, old_probs: torch.Tensor, rewards: torch.Tensor, gamma: float):
##        """
##        Optimize the policy network.
##        """
##        # Compute advantages
##        advantages = self.compute_advantages(rewards, states, gamma)
##
##        # Compute new probabilities
##        logits = self.policy(states)
##        new_probs = Categorical(logits=logits).probs.gather(1, actions.unsqueeze(1))
##
##        # Compute the surrogate loss
##        loss = self.surrogate_loss(old_probs, new_probs, advantages)
##
##        # Compute KL divergence
##        kl = self.compute_kl(old_probs, new_probs)
##
##        # Gradient computation
##        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
##        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
##
##        # Compute the Fisher Information Matrix (FIM)
##        fim = torch.autograd.grad(flat_grad_kl, self.policy.parameters(), create_graph=True)
##        flat_fim = torch.cat([grad.view(-1) for grad in fim])
##
##        # Compute the inverse of FIM
##        inv_fim = torch.linalg.pinv(flat_fim)
##
##        # Compute the natural gradient
##        natural_grad = torch.matmul(inv_fim, flat_grad_kl)
##
##        # Compute the step direction
##        step_dir = torch.sign(natural_grad) * torch.sqrt(2 * self.max_kl)
##
##        # Update policy parameters
##        new_params = torch.cat([param.view(-1) for param in self.policy.parameters()]) + step_dir
##        self.update_policy_parameters(new_params)
##
##        # Optimize value network
##        self.update_value_network(states, rewards)
##
##    def update_value_network(self, states: torch.Tensor, rewards: torch.Tensor):
##        """
##        Update the value network.
##        """
##        predicted_values = self.value(states)
##        value_loss = nn.MSELoss()(predicted_values, rewards)
##        self.value_optimizer.zero_grad()
##        value_loss.backward()
##        self.value_optimizer.step()
##        
##    def compute_kl(self, old_probs, new_probs):
##        return (old_probs * torch.log(old_probs / new_probs)).sum()

##    def update_policy_parameters(self, new_params):
##        idx = 0
##        for param in self.policy.parameters():
##            param_length = param.nelement()
##            param.data = new_params[idx:idx + param_length].view(param.size())
##            idx += param_length
##
##    def optimize_policy(self, states, actions, old_probs, rewards, gamma):
##        advantages = self.compute_advantages(rewards, states, gamma)
##        logits = self.policy(states)
##        new_probs = Categorical(logits=logits).probs.gather(1, actions.unsqueeze(1))
##
##        loss = self.surrogate_loss(old_probs, new_probs, advantages)
##        kl = self.compute_kl(old_probs, new_probs)
##        
##        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
##        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
##        
##        fim = torch.autograd.grad(flat_grad_kl, self.policy.parameters(), create_graph=True)
##        flat_fim = torch.cat([grad.view(-1) for grad in fim])
##        
##        inv_fim = torch.linalg.pinv(flat_fim)
##        natural_grad = torch.matmul(inv_fim, flat_grad_kl)
##        
##        step_dir = torch.sign(natural_grad) * torch.sqrt(2 * self.max_kl)
##        
##        new_params = torch.cat([param.view(-1) for param in self.policy.parameters()]) + step_dir
##        self.update_policy_parameters(new_params)
##
##        self.update_value_network(states, rewards)
##
##    def update_value_network(self, states, rewards):
##        predicted_values = self.value(states)
##        value_loss = nn.MSELoss()(predicted_values, rewards)
##        self.value_optimizer.zero_grad()
##        value_loss.backward()
##        self.value_optimizer.step()

# Implement the rest of the code such as training loop, environment interactions, etc.

# Example usage
##input_dim = 5
##output_dim = 2
##policy_network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
##value_network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
##optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
##value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##trpo = TRPO(policy_network, value_network, optimizer, value_optimizer)


from itertools import product
from tqdm import tqdm

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


class BayesianTRPOValue:
    def __init__(self, input_dim, hidden_dim, output_dim, environment, delta=0.01, gamma=0.99, lam=0.95):
        self.policy = BayesianPolicy(input_dim, hidden_dim, output_dim)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro.optim as pyro_optim

class BayesianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianPolicy, self).__init__()
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
        # Implement your ADVI logic.
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


from scipy.stats import beta
import numpy as np

# Your BayesianSharpeRatio class here ...

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
        self.optimal_bet_size = 0.0  # Initialize optimal bet size
    
    def reset(self):
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.bayesian_sharpe = BayesianSharpeRatio()
        self.optimal_bet_size = 0.0
        return self._get_observation()
    
    def _get_observation(self):
        # Implement how you want to represent the state
        return np.array([])  # Dummy
    
    def step(self, action):
        self.current_step += 1
        self._execute_trade(action)
        
        next_obs = self._get_observation()
        reward = self.calculate_reward()
        done = self.current_step >= len(self.data) - 1  # Assuming self.data is a time-series data
        
        return next_obs, reward, done
    
    def _execute_trade(self, action):
        # Implement your logic for executing a trade given an 'action'
        pass
    
    def calculate_reward(self):
        daily_return = self.returns[-1] if self.returns else 0.0  # Assume daily return for the last step
        self.bayesian_sharpe.update(daily_return)
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        
        # Calculate reward, possibly considering optimal_bet_size
        reward = 0.0  # Implement your custom reward
        return reward
    
    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, BTC: {self.btc_quantity}, Reward: {self.calculate_reward()}")



    from scipy.stats import beta
import numpy as np
import random

# Your BayesianSharpeRatio class definition should remain unchanged.





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
        self.optimal_bet_size = 0.0
    
    def reset(self):
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.bayesian_sharpe = BayesianSharpeRatio()
        self.optimal_bet_size = 0.0
        return self._get_observation()



    def _get_observation(self):
        # For simplicity, we'll just return the current BTC price as the state.
        # In a real-world scenario, this could be more complex.
        return np.array([self.data[self.current_step]['BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv']])
    
    def step(self, action):
        self.current_step += 1
        self._execute_trade(action)
        
        next_obs = self._get_observation()
        reward = self.calculate_reward()
        done = self.current_step >= len(self.data) - 1
        
        return next_obs, reward, done
    
    def _execute_trade(self, action):
        current_price = self.data[self.current_step]['BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv']
        if action == 0:  # Buy
            self.btc_quantity += (self.balance / current_price) * (1 - self.transaction_fee)
            self.balance = 0.0
        elif action == 1:  # Hold
            pass
        elif action == 2:  # Sell
            self.balance += (self.btc_quantity * current_price) * (1 - self.transaction_fee)
            self.btc_quantity = 0.0
        
        self.historical_values.append(self.balance + self.btc_quantity * current_price)
        if len(self.historical_values) > 1:
            daily_return = (self.historical_values[-1] / self.historical_values[-2]) - 1
            self.returns.append(daily_return)
    
    def calculate_reward(self):
        daily_return = self.returns[-1] if self.returns else 0.0
        self.bayesian_sharpe.update(daily_return)
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        
        # For the reward, we'll simply return the daily_return scaled by the optimal bet size.
        reward = daily_return * self.optimal_bet_size
        return reward
    
    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, BTC: {self.btc_quantity}, Reward: {self.calculate_reward()}")


from scipy.stats import beta
import numpy as np
import random

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
        self.optimal_bet_size = 0.0
    
    def reset(self):
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.returns = []
        self.historical_values = []
        self.btc_quantity = 0.0
        self.bayesian_sharpe = BayesianSharpeRatio()
        self.optimal_bet_size = 0.0
        return self.get_state()

    def _get_observation(self):
        return np.array([self.data[self.current_step]['BTC: Realized Price_bitcoin-14-day-market-realised-gradient.csv']])
    
    def step(self, action):
        self.current_step += 1
        self.execute_trade(action)
        
        next_obs = self.get_state()
        reward = self.calculate_reward()
        done = self.current_step >= len(self.data) - 1
        
        return next_obs, reward, done
    
    def calculate_reward(self):
        daily_return = self.returns[-1] if self.returns else 0.0
        self.bayesian_sharpe.update(daily_return)
        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
        reward = daily_return * self.optimal_bet_size
        return reward

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, BTC: {self.btc_quantity}, Reward: {self.calculate_reward()}")


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

        self.historical_values.append(self.balance + self.btc_quantity * price)
        if len(self.historical_values) > 1:
            daily_return = (self.historical_values[-1] / self.historical_values[-2]) - 1
            self.returns.append(daily_return)
            
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

##    def step(self, action):
####        self.dynamic_bet_sizing()  # Calculate the optimal bet size dynamically
####        scaled_action = action * self.optimal_bet_size  # Scale the action by the optimal bet size
##        
##        self.current_step += 1
##        self.optimal_bet_size = self.bayesian_sharpe.calculate_kelly_bet()
####        self.execute_trade(action * self.optimal_bet_size)  # Scale action by optimal bet size
##        
##        print('optimal_bet_size: ', self.optimal_bet_size)
##        
####        if self.current_step >= len(self.data):
####            return self.reset(), 0, True
##
##        self.execute_trade(action)
##        
##        reward = self.calculate_reward()
##        print("Reward:", reward)  # Debugging line
##        self.returns.append(reward.item() if torch.is_tensor(reward) else float(reward))
##        
##        done = self.current_step >= len(self.data) - 1
##        
##        return self.get_state(), reward, done



class TRPO:
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
                states_tensor = torch.FloatTensor(state)
                action_prob_means = self.policy.model(states_tensor)
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
        
##        policy_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
##        value_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))
####        policy_network = BayesianPolicy(state_dim, 64, 1)
####        value_network = BayesianTRPOValue(state_dim, 64, 1, env)
##        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
##        value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##        trpo = TRPO(policy_network, value_network, optimizer, value_optimizer)


##        trpo = SACAgent(state_dim, action_dim, hidden_dim, learning_rate, entropy_weight)
        
        logging.info(f'state dim = {state_dim}')


        logging.info(f'state dim = {state_dim}')









        ### Example usage
        input_dim = len(env.get_state())
        action_dim = 1
        hidden_dim = 32
        hidden2_dim = 16
        output_dim = 1

        # Initialize Environment, Policy, and Value network
        # env = YourEnvironmentHere()
##        policy_net = BayesianTRPOPolicy(input_dim, hidden_dim, output_dim)
##        value_net = BayesianTRPOValue(input_dim, hidden_dim, 1, env)
##
##        # Initialize TRPO
##        trpo = TRPO(policy_net, value_net)

        policy_network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        policy_network = BayesianPolicy(input_dim, 64, output_dim)
        
        value_network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        value_network = BayesianPolicy(input_dim, 64, output_dim)
        
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
        value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##        optimizer = optim.Adam(value_network.parameters(), lr=0.001)
##        trpo = TRPO(policy_network, value_network, optimizer, value_optimizer)
        trpo = TRPO(policy_network, value_network, env)
####        trpo = TRPO(input_dim, output_dim, value_network, optimizer, value_optimizer)
        states_tensor = torch.FloatTensor(state)

        trpo.train()

##        # Initialize your policy and value networks, then the TRPO agent
##        policy_net = policy_network
##        value_net = value_network
##        trpo_agent = TRPO(policy_net, value_net)

##    for epoch in range(num_epochs):
##        for batch in data_loader:
##            # ... TRPO training code here
##            states_tensor = torch.FloatTensor(states)

##        # Training loop
##        for episode in range(100):
##            state = env.reset()
##            states, actions, rewards = trpo.sample_trajectory(env, state, T=1000)
##            print(states)
##            advantages = trpo.compute_advantages(rewards, states, gamma=0.5)
##            print(advantages)
##
##            trpo.update_policy(states, actions, advantages)




        # Training loop
        n_episodes = num_episodes

        # Initialize variables outside the loop
        states_list = []

        # Main loop
        with tqdm(total=n_episodes) as pbar:
            for episode in tqdm(range(n_episodes)):
                pbar.set_postfix({"Episode": f"{episode+1}/{n_episodes}"}, refresh=True)
                
                state = env.reset()
                done = False
                episode_reward = 0
                actions = []
                
                while not done:
                    # Let's make sure we have a numpy array, not a list
                    single_state_np = np.array(state)
                    states_tensor = torch.FloatTensor(single_state_np).unsqueeze(0)  # Convert to tensor and add a batch dimension
                    
                    # Sample the trajectory
                    _, actions, rewards = trpo.sample_trajectory(env, state, T=3)
                    
                    states_list.append(single_state_np)  # Append the state to the states list
                    
                    # Compute advantages
                    advantages = trpo.compute_advantages(rewards, states_tensor, gamma=0.99)
                    print(f"Advantages: {advantages}")

                # Outside the inner while loop, convert the list of states into a tensor
                states_array = np.array(states_list)
                states_tensor = torch.FloatTensor(states_array)
                
                # Now update the policy
                trpo.update_policy(states_tensor, actions, advantages)



        
##        with tqdm(total=n_episodes) as pbar:
##            for episode in tqdm(range(n_episodes)):
##                pbar.set_postfix({"Episode": f"{episode+1}/{n_episodes}"}, refresh=True)
##                
##                state = env.reset()
##                done = False
##                episode_reward = 0
##                actions = []
##                while not done:
##
####                    states, actions, rewards = trpo.sample_trajectory(env, state, T=1000)
####                    states_tensor = torch.tensor(states, dtype=torch.float32) # Convert states to tensor
####                    advantages = trpo.compute_advantages(rewards, states_tensor, gamma=0.99)
##
##                    
##                    states_tensor = torch.FloatTensor(state)
##                    states_tensor, actions, rewards = trpo.sample_trajectory(env, state, T=1000)
##                    print(state)
##                    advantages = trpo.compute_advantages(rewards, states_tensor, gamma=0.99)
##                    print(advantages)
##
##                trpo.update_policy(states, actions, advantages)

# Main function
if __name__ == '__main__':
    main()

        
'''
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
        
        policy_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        value_network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
        value_optimizer = optim.Adam(value_network.parameters(), lr=0.001)
        trpo = TRPO(policy_network, value_network, optimizer, value_optimizer)


##        trpo = SACAgent(state_dim, action_dim, hidden_dim, learning_rate, entropy_weight)
        
        logging.info(f'state dim = {state_dim}')
        # Initialize environment and agent
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
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    

                    # Get action
                    with torch.no_grad():
##                        action_mean, action_std = trpo.actor(state_tensor)
##                        action_distribution = Normal(action_mean, action_std)
##                        sampled_action = action_distribution.sample()
##                        tanh_action = torch.tanh(sampled_action)
                        
                        action_mean, action_std = trpo.actor(state_tensor)
                        action_distribution = torch.distributions.Normal(action_mean, action_std)
                        print('action is action_distribution :', action_distribution)
                        sampled_action = action_distribution.sample()
                        print('action is after sampled_action :', sampled_action)

                        action = torch.tanh(action_distribution.sample()) 
                        print('action is mains get action :', action)
                        
##                    print(actions)

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
                    trpo.update(state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            
                action_mean, action_std = trpo.actor(state_tensor)
                action_std = torch.clamp(action_std, min=1e-3)
                action_distribution = torch.distributions.Normal(action_mean, action_std)
                pbar.set_postfix({"Episode Reward": episode_reward}, refresh=True)

        # Compute episode reward, update best_hyperparams and best_reward if needed
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_hyperparams = hyperparams
            
    print(f'Best hyperparameters: {best_hyperparams}')
    print(f'Best reward: {best_reward}')


# Main function
if __name__ == '__main__':
    main()
'''
        
