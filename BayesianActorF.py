# Let's start with integrating ExtendedAdvancedKellyCriterion into the BayesianActor class

import numpy as np
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
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)

class BayesianPolicyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianPolicyNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.cov_matrix = None  # Placeholder for covariance matrix
        self.prior_std = 1.0  # Placeholder for prior standard deviation


    def update_cov_matrix(self):
        """
        Method to update the covariance matrix based on the BNN's current state.
        This is a placeholder and can be implemented based on specific requirements.
        """
        self.cov_matrix = torch.eye(3)  # Example: Identity matrix, this would be calculated based on BNN weights
        return self.cov_matrix

    def adjust_priors(self, risk_level):
        """
        Adjust the priors of the BNN based on the risk level received from Advanced Kelly Criterion.
        """
        self.prior_std = risk_level / 10.0  # Example: Setting the prior std proportional to the risk level

class AdvancedKellyCriterionContinuous:
    def __init__(self):
        self.risk_level = None  # Placeholder for risk level
        self.action_limit_factor = 1.0  # Placeholder for action range limiting factor

    def calculate_risk_level(self, cov_matrix):
        """
        Calculate the risk level based on the covariance matrix.
        """
        self.risk_level = torch.trace(cov_matrix).item()
        return self.risk_level

    def adjust_action_limit(self):
        """
        Adjust the range of possible actions based on the calculated risk level.
        For demonstration, if risk level > 10, reduce the action range (e.g., from [-1, 1] to [-0.5, 0.5]).
        """
        if self.risk_level > 10.0:
            self.action_limit_factor = 0.5
        else:
            self.action_limit_factor = 1.0
        return self.action_limit_factor

    def adjust_exploration(self, base_std_dev):
        """
        Adjust the exploration strategy based on the risk level.
        For demonstration, if risk level > 10, reduce the standard deviation of Gaussian noise added to actions.
        """
        if self.risk_level > 10.0:
            return base_std_dev * 0.5
        else:
            return base_std_dev

    def update_cov_matrix(self, cov_matrix):
        """
        Update the covariance matrix received from BayesianPolicyNN and 
        adjust risk level accordingly.
        """
        self.risk_level = cov_matrix.trace()  # Example: using trace of the covariance matrix as risk level
        return self.risk_level


class ExtendedAdvancedKellyCriterion(AdvancedKellyCriterionContinuous):
    def __init__(self):
        super(ExtendedAdvancedKellyCriterion, self).__init__()
        self.entropy_bonus = 0.0  # Placeholder for entropy bonus
        self.epistemic_uncertainty = 0.0  # Placeholder for epistemic uncertainty

    def calculate_entropy_bonus(self, action_prob_distribution):
        """
        Calculate the entropy bonus based on the action probability distribution.
        """
        self.entropy_bonus = -torch.sum(action_prob_distribution * torch.log(action_prob_distribution))
        return self.entropy_bonus

    def calculate_epistemic_uncertainty(self, value_distribution):
        """
        Calculate the epistemic uncertainty based on the value distribution (from Bayesian Critic).
        """
        # Using the variance of the value distribution as a measure of epistemic uncertainty
        self.epistemic_uncertainty = torch.var(value_distribution)
        return self.epistemic_uncertainty

    def overall_risk_assessment(self):
        """
        Combine various metrics to perform an overall risk assessment.
        For demonstration, we'll sum the calculated risk level, entropy bonus, and epistemic uncertainty.
        """
        overall_risk = self.risk_level + self.entropy_bonus + self.epistemic_uncertainty
        return overall_risk

    
class BayesianActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, advanced_kelly):
        super(BayesianActor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.advanced_kelly = advanced_kelly  # Use the passed instance
        
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action = self.layer3(x)
        return action

    def sample_action(self, state, base_std_dev=0.2):
        """
        Sample an action based on the current policy and the overall risk level.
        """
        # Get the mean action from the network
        mean_action = self.forward(state)

        # Adjust the standard deviation based on the overall risk level
        adjusted_std_dev = self.advanced_kelly.adjust_exploration(base_std_dev)

        # Create a normal distribution with the mean action and adjusted standard deviation
        action_distribution = torch.distributions.Normal(mean_action, adjusted_std_dev)

        # Sample an action from this distribution
        sampled_action = action_distribution.rsample()

        return sampled_action







# Example usage (assuming you've loaded the ExtendedAdvancedKellyCriterion class and other necessary modules)
bnn = BayesianPolicyNN(input_dim=10, hidden_dim=20, output_dim=2)

# Mock state tensor for demonstration
state = torch.Tensor([1.0, 0.5, 0.2, -0.5, 0.3, 0.6, -0.1, 0.7, -0.2, 0.1])

# Create the AdvancedKellyCriterion instance
ekc = ExtendedAdvancedKellyCriterion()

# Create the AdvancedKellyCriterion instance
akc = AdvancedKellyCriterionContinuous()

# Initialize the risk_level immediately (This is the new line)
akc.update_cov_matrix(torch.eye(3))  # Using an identity matrix as a placeholder

# Create the BayesianActor instance and pass it the AdvancedKellyCriterion instance
actor = BayesianActor(input_dim=10, hidden_dim=20, output_dim=2, advanced_kelly=akc)

# Calculate the covariance matrix
cov_matrix = bnn.update_cov_matrix()

### Update the risk level in AdvancedKellyCriterionContinuous
##risk_level = akc.update_cov_matrix(cov_matrix)
### Update the risk level in ExtendedAdvancedKellyCriterion (if used)
# Calculate the covariance matrix
cov_matrix = bnn.update_cov_matrix()

# Update the risk level in AdvancedKellyCriterion
risk_level = akc.update_cov_matrix(cov_matrix)  # Make sure this line is before sampling an action

# Print for debugging
print("Updated Covariance Matrix: ", cov_matrix)
print("Updated Risk Level: ", risk_level)

# Sample an action based on this state
sampled_action = actor.sample_action(state)

print("Sampled Action: ", sampled_action)

