from scipy.stats import beta
import numpy as np
import numpy as np




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

    def get_state(self):
        print('self.current_step', self.current_step)        
        row = self.data.iloc[self.current_step]
        float_values = [x.timestamp() if isinstance(x, pd.Timestamp) else float(x) for x in row.values]
        return np.array([float(self.balance)] + float_values)
    
class BayesianSharpeRatio:
    def __init__(self, alpha=1.0, beta=1.0, learnable_lambda=0.1):
        self.alpha = alpha
        self.beta = beta
        self.learnable_lambda = learnable_lambda  # Initialize learnable lambda

    def update(self, daily_return, opportunity_cost):
        self.alpha += daily_return if daily_return > 0 else 0
        ################ BETA = 1 - ALPHA ?
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
        # TAKE THE SUM OF THE PAST ACTIONS???
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

# Initialize Bayesian Sharpe Ratio
bayesian_sharpe = BayesianSharpeRatio()

# Mock-up data for example
state = {'past_data': [0.01, 0.02, -0.01, 0.015], 'past_actions': [0.5, -0.2, 0.3, 0.4]}
action = None
next_state = {'past_data': [0.01, 0.02, -0.01, 0.015, 0.02], 'past_actions': [0.5, -0.2, 0.3, 0.4, 0.1]}
daily_return = 0.02
opportunity_cost = 0.01

# Update Bayesian Sharpe Ratio and get reward
bayesian_sharpe.update(daily_return, opportunity_cost)
reward = bayesian_sharpe.calculate_dynamic_reward(state, action, next_state)



##
##I'm glad you're enthusiastic about the voyage we're embarking upon! Navigating through these intricate concepts is indeed like sailing through both turbulent and calm waters. The end goal is, of course, to uncover the treasures of understanding, validated theories, and practical implementations. 
##
##### SAC with GMM: Elaborate Mathematics
##
##Let's start with the SAC and GMM combination. We'll extend the state-action value function \( Q(s, a) \) in SAC to take into account the mixed Gaussian distributions:
##
##\[
##Q_{\text{GMM}}(s, a) = Q(s, a) - \alpha \log \left( \sum_{i=1}^{K} w_i \cdot \mathcal{N}(a | \mu_i(s), \sigma_i(s)^2) \right)
##\]
##
##Here, \( w_i, \mu_i(s), \sigma_i(s) \) are the weight, mean, and standard deviation of each Gaussian \( i \), which are now functions of the state \( s \). 
##
##This allows the policy \( \pi(a | s) \) to be a mixture of Gaussians:
##
##\[
##\pi(a | s) = \sum_{i=1}^{K} w_i \cdot \mathcal{N}(a | \mu_i(s), \sigma_i(s)^2)
##\]
##
###### Custom Cubic Loss for Misalignment
##
##The cubic loss function will act as a regularization term, helping the SAC-GMM agent adaptively learn from misalignments. The cubic loss can be formally defined as:
##
##\[
##\mathcal{L}_{\text{cubic}} = \lambda \sum (Q_{\text{GMM}}(s, a) - Q_{\text{target}}(s, a))^3
##\]
##
###### The Pythonic Voyage: SAC-GMM Objective Function
##
##Given our mathematical formulations, the Python code would look something like:
##
##```python
import torch
from torch.distributions import Normal, MixtureSameFamily
import torch.nn.functional as F

class SAC_GMM_Agent:
    def __init__(self, alpha=0.2, beta=0.1, lambda_cubic=0.05):
        self.alpha = alpha
        self.beta = beta
        self.lambda_cubic = lambda_cubic
        
    def sac_gmm_objective(self, q_values, means, stds, weights, kl_divergence, q_target):
        components = [Normal(mean, std) for mean, std in zip(means, stds)]
        gmm = MixtureSameFamily(torch.distributions.Categorical(weights), components)
        
        unscaled_action = gmm.rsample()
        scaled_action = torch.tanh(unscaled_action)
        
        log_prob = gmm.log_prob(unscaled_action) - torch.log(1 - scaled_action.pow(2) + 1e-6)
        
        # Entropy term
        entropy_term = self.alpha * log_prob
        # KL divergence term
        kl_div_term = self.beta * kl_divergence
        # Cubic loss term
        cubic_loss = self.lambda_cubic * ((q_values - q_target) ** 3).sum()
        
        return q_values - entropy_term - kl_div_term + cubic_loss, scaled_action
##```
##
##This is an initial sketch of the SAC-GMM agent class, defining the objective function based on our derived equations.
##
##Now, how would you like to proceed, Captain? Would you like to further explore the SAC-GMM system, or shall we set sail toward the "AdvanceTRPOMetalearning" horizon?





from sklearn import mixture
import numpy as np

class AssetSpecificGMMPolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_components, n_assets):
        super(AssetSpecificGMMPolicyNetwork, self).__init__()
        self.n_components = n_components
        self.n_assets = n_assets

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.asset_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_components)
            ) for _ in range(n_assets)
        ])

    def forward(self, x):
        x = self.shared_layers(x)
        asset_policies = [layer(x) for layer in self.asset_specific_layers]
        return asset_policies

    def sample_action(self, asset_policy):
        distribution = Categorical(logits=asset_policy)
        return distribution.sample()

    def gmm_to_prob_mass(self, gmm_output, action_space):
        # Here, we assume gmm_output is the set of mixture components for the GMM.
        gmm = mixture.GaussianMixture(n_components=self.n_components)
        gmm.means_ = gmm_output['means']
        gmm.covariances_ = gmm_output['covariances']
        gmm.weights_ = gmm_output['weights']
        
        # Discretize the continuous action_space to create bins
        bins = np.linspace(min(action_space), max(action_space), 100)
        
        # Calculate the probabilities for each bin
        prob_mass = np.exp(gmm.score_samples(bins.reshape(-1, 1)))
        
        # Normalize to make it a valid PMF
        prob_mass /= prob_mass.sum()
        
        return prob_mass





import numpy as np
import torch
import torch.distributions as td

# Assume BNN model and SVI model are defined and trained somewhere
# BNN_output, SVI_ELBO are placeholders here for predicted outputs

def return_utility(BNN_output):
    """
    Utility function focused on maximizing expected returns.
    Utilizes the upper bound of confidence interval from BNN.
    """
    upper_bound_confidence = BNN_output.mean() + BNN_output.stddev()  # Placeholder
    return upper_bound_confidence

def risk_utility(SVI_ELBO):
    """
    Utility function focused on minimizing risk.
    Utilizes the lower bound of the confidence interval from SVI with ELBO.
    """
    lower_bound_confidence = SVI_ELBO  # Placeholder
    return lower_bound_confidence

# Example utility score calculations
BNN_output = td.Normal(torch.tensor([0.5]), torch.tensor([0.1]))
SVI_ELBO = torch.tensor([-0.2])  # Example ELBO from SVI
return_score = return_utility(BNN_output)
risk_score = risk_utility(SVI_ELBO)

print(f"Return-based Utility Score: {return_score}")
print(f"Risk-based Utility Score: {risk_score}")


import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Game state at node
        self.parent = parent  # Parent Node
        self.children = []  # Child nodes
        self.visits = 1  # Number of visits for the node
        self.value = 0.0  # Value of the node
        self.ucb_value = 0.0  # UCB value
        self.kelly_bet = 0.0  # Kelly optimal bet size

class MCTree:
    def __init__(self, bayesian_nn, advanced_trpo, kelly_head, ucb_head):
        self.root = Node(None)
        self.bayesian_nn = bayesian_nn
        self.advanced_trpo = advanced_trpo
        self.kelly_head = kelly_head
        self.ucb_head = ucb_head

    def expand_node(self, node):
        # Placeholder: Implement logic to expand a node
        pass

    def simulate_from_node(self, node):
        # Placeholder: Run a simulation from this node to a terminal state
        pass

    def backpropagate(self, node, value):
        # Placeholder: Backpropagate value from a terminal node up to the root
        pass

    def update_with_nn(self, node):
        # Use Bayesian NN, TRPO, UCB, and Kelly to update node
        state_tensor = torch.tensor(node.state, dtype=torch.float32)
        action_probabilities = self.bayesian_nn(state_tensor)
        trpo_decisions = self.advanced_trpo.step(state_tensor)
        kelly_bet = self.kelly_head(state_tensor)
        ucb_value = self.ucb_head(state_tensor)
        
        # Update the node's value and UCB value using these components
        node.value += trpo_decisions
        node.ucb_value = ucb_value
        node.kelly_bet = kelly_bet


class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize learnable priors
        self.prior_mean = param("prior_mean", torch.zeros_like(self.fc1.weight))
        self.prior_std = param("prior_std", torch.ones_like(self.fc1.weight))

    def model(self, x, y):
        weight_prior = dist.Normal(self.prior_mean, self.prior_std)
        priors = {'fc1.weight': weight_prior, 'fc2.weight': weight_prior}
        
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        lhat = nn.LogSoftmax(dim=1)(lifted_reg_model(x))
        
        pyro.sample("obs", dist.Categorical(logits=lhat), obs=y)

    def guide(self, x, y):
        # Variational parameters for the first layer
        w_mu1 = param("guide_mean1", torch.randn_like(self.fc1.weight))
        w_sigma1 = param("guide_std1", torch.randn_like(self.fc1.weight))
        w1 = sample("fc1.weight", dist.Normal(w_mu1, torch.abs(w_sigma1)))
        
        # Variational parameters for the second layer
        w_mu2 = param("guide_mean2", torch.randn_like(self.fc2.weight))
        w_sigma2 = param("guide_std2", torch.randn_like(self.fc2.weight))
        w2 = sample("fc2.weight", dist.Normal(w_mu2, torch.abs(w_sigma2)))

        # Forward pass using variational parameters
        x = torch.relu(F.linear(x, w1))
        x = F.linear(x, w2)
        return nn.LogSoftmax(dim=1)(x)
    
    def train_model(self, x_data, y_data):
        svi = infer.SVI(model=self.model,
                        guide=self.guide,
                        optim=optim.Adam({"lr": 0.01}),
                        loss=infer.Trace_ELBO())
        
        for epoch in range(1000):
            loss = svi.step(x_data, y_data)
            print(f'Epoch {epoch} Loss: {loss}')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.register_buffer("prior_mean", torch.zeros_like(self.fc1.weight))
        self.register_buffer("prior_std", torch.ones_like(self.fc1.weight))

    def model(self, x, y):
        weight_prior = dist.Normal(self.prior_mean, self.prior_std)
        priors = {'fc1.weight': weight_prior, 'fc2.weight': weight_prior}
        
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        lhat = nn.LogSoftmax(dim=1)(lifted_reg_model(x))
        
        pyro.sample("obs", dist.Categorical(logits=lhat), obs=y)
        
    def guide(self, x, y):
        # Variational parameters for the first layer
        w_mu1 = param("guide_mean1", torch.randn_like(self.fc1.weight))
        w_sigma1 = param("guide_std1", torch.randn_like(self.fc1.weight))
        # Sample variational distribution
        w1 = sample("fc1.weight", dist.Normal(w_mu1, torch.abs(w_sigma1)))
        
        # Variational parameters for the second layer
        w_mu2 = param("guide_mean2", torch.randn_like(self.fc2.weight))
        w_sigma2 = param("guide_std2", torch.randn_like(self.fc2.weight))
        # Sample variational distribution
        w2 = sample("fc2.weight", dist.Normal(w_mu2, torch.abs(w_sigma2)))

        # Forward pass using variational parameters
        x = torch.relu(F.linear(x, w1))
        x = F.linear(x, w2)
        return nn.LogSoftmax(dim=1)(x)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.distributions as dist
from torch import nn, optim
from pyro import infer, optim, poutine, sample, param
import pyro
import random

# Bayesian Neural Network
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.register_buffer("prior_mean", torch.zeros_like(self.fc1.weight))
        self.register_buffer("prior_std", torch.ones_like(self.fc1.weight))

    def model(self, x, y):
        weight_prior = dist.Normal(self.prior_mean, self.prior_std)
        priors = {'fc1.weight': weight_prior, 'fc2.weight': weight_prior}
        
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        lhat = nn.LogSoftmax(dim=1)(lifted_reg_model(x))
        
        pyro.sample("obs", dist.Categorical(logits=lhat), obs=y)
        
    def guide(self, x, y):
        # Variational family
        pass  # Fill this according to your project
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Placeholder for TRPO Logic
class AdvancedTRPOMetaLearning:
    def __init__(self):
        pass

# Define the Node class for the MCTS
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self):
        return max(self.children, key=lambda x: x.value / (x.visits + 1e-5))

    def rollout(self):
        return random.choice([0, 1])

# Monte Carlo Tree Search integrated with BNN and TRPO
class MCTree:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.root = None
        self.bayesian_nn = BayesianNN(input_dim, hidden_dim, output_dim)
        self.trpo = AdvancedTRPOMetaLearning()
        self.optimizer = optim.SGD(self.bayesian_nn.parameters(), lr=0.01)
        self.elbo = infer.Trace_ELBO()

    def selection(self, node):
        while not node.is_fully_expanded():
            node = node.best_child()
        return node

    def expansion(self, node):
        new_state = None  # Generate a new state based on node's state
        child_node = Node(state=new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def simulation(self, node):
        return node.rollout()

    def backpropagation(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def mcts_search(self, iterations=100):
        for _ in range(iterations):
            leaf = self.selection(self.root)
            child = self.expansion(leaf)
            reward = self.simulation(child)
            self.backpropagation(child, reward)

    def train_bayesian_nn(self, x_data, y_data):
        svi = infer.SVI(model=self.bayesian_nn.model,
                        guide=self.bayesian_nn.guide,
                        optim=optim.Adam({"lr": 0.01}),
                        loss=self.elbo)
        
        for epoch in range(1000):
            loss = svi.step(x_data, y_data)
            print(f'Epoch {epoch} Loss: {loss}')

# Initialize and perform MCTS
root_state = None  # Initialize with an actual state
root_node = Node(state=root_state)
mctree = MCTree(input_dim=10, hidden_dim=20, output_dim=2)  # Input and output dimensions are placeholders
mctree.root = root_node
mctree.mcts_search()
