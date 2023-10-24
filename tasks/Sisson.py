import torch
import torch.distributions as D # for distributions
from sbi import utils as sutils

# hyperparams
theta_dim = x_dim = 3
amp = 5
sig_true = 0.7*torch.ones(theta_dim,theta_dim) + 0.3*torch.eye(theta_dim)
w_mix = 0.3

# prior
prior = sutils.BoxUniform(low=-15*torch.ones(theta_dim), high=15*torch.ones(theta_dim))

# simulator
def simulator(mus):
    n = mus.shape[0]
    shape = mus.shape[1] 
    b = torch.bernoulli(w_mix*torch.ones(n,shape))
    Mus = (1-2*b)*mus
    eps = D.MultivariateNormal(loc=torch.zeros(shape), covariance_matrix=sig_true).sample((n,))
    return Mus + eps

# surrogate prior
sur_prior = sutils.BoxUniform(low=-15*torch.ones(x_dim), high=15*torch.ones(x_dim))

# different budgets tested
samples_len = [250, 500, 1000, 1500, 2500, 5000, 7500, 10000]

# x_obs
x_obs = amp*torch.ones(1, x_dim)

# true_sample
# Since the prior is uniform and very large (i.e., $\propto 1$) the posterior is $\propto p(x|\theta)$, i.e. the posterior is simply the likelihood with the $x$ and $\theta$ changing roles. As such we can draw from the likelihood (the simulator) and treat that sample as a sample from the true posterior. 
mus = x_obs.repeat(2000,1)
true_sample = simulator(mus)

