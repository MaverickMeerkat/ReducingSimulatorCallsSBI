import torch
import torch.distributions as D # for distributions
from sbi import utils as sutils

# hyperparams
theta_dim = 1
dim = dim_theta = 6
dim_x = x_dim = 10
n = 2000
sig = 0.1

# # generate the same x's (u's) that will be used for all different coefficients (theta/beta)
# x = D.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta)).sample((dim_x,))
# torch.save(x, f"{path}/x.pkl")
x = torch.load("./saved/x.pkl")
# theta_true = prior.sample((1,))
# y_obs = simulator(theta_true)
# torch.save(y_obs, f"{path}/y_obs.pkl")
x_obs = y_obs = torch.load("./saved/y_obs.pkl")
# torch.save(theta_true, f"{path}/theta_true.pkl")
theta_true = torch.load("./saved/theta_true.pkl")

# prior
prior = D.MultivariateNormal(torch.zeros(dim_theta), torch.eye(dim_theta))

# simulator
def simulator(theta):
  n = theta.shape[0]
  y = theta @ x.T + noise.sample((n, dim_x))
  return y

# surrogate prior (not really needed, as not run sequentially) 
sur_prior = sutils.BoxUniform(low=-10*torch.ones(x_dim), high=10*torch.ones(x_dim)) 

# different budgets tested
samples_len = [100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000]

# true_sample
lambd = 100*(x.T @ x) + 1*torch.eye(dim_theta)
Sigma = torch.inverse(lambd)
mu = 100*Sigma @ x.T @ y_obs.T
true_posterior = D.MultivariateNormal(mu.squeeze(), Sigma)
true_sample = true_posterior.sample((2000,))
