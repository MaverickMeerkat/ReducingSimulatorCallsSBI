import torch
import torch.distributions as D # for distributions
from sbi import utils as sutils

# hyperparams
theta_dim = 5
x_dim = 16

# prior
lower = -3
upper = 3
prior = sutils.BoxUniform(low=lower*torch.ones(theta_dim), high=upper*torch.ones(theta_dim))

# simulator
def simulator(theta):
    n = theta.shape[0] 
    mvn = torch.zeros(n,16)
    for i in range(n):
      mu = theta[i,:2]
      s1 = theta[i,2]**2
      s2 = theta[i,3]**2
      r = torch.tanh(theta[i,4])
      S = torch.tensor([[s1**2, r*s1*s2],[r*s1*s2, s2**2]])
      mvn[i,:] = D.MultivariateNormal(loc=mu, covariance_matrix=S).sample((8,)).flatten()
    return mvn

# surrogate prior (took rough limits of x's generated by prior theta's) 
sur_prior = sutils.BoxUniform(low=-30*torch.ones(16), high=30*torch.ones(16))

# different budgets tested
samples_len = [250, 500, 1000, 2500, 5000, 7500, 10000]

# x_obs
x_obs = torch.load("./saved/SLCP_8points_x_obs.pt")

# true_sample - check "SLCP - generate true posterior.ipynb"
true_sample_full = torch.load("./saved/SLCP_sample_8points_mcmc.pt")
p = torch.full((1,100000), fill_value=1/100000)
idx = p.multinomial(num_samples=10000, replacement=False)
ids = idx.squeeze()
true_sample_red = true_sample_full[ids,:]
true_sample = true_sample_red
