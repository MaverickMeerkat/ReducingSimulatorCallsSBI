import torch
import torch.distributions as D # for distributions
from sbi import utils as sutils

# hyperparams
theta_dim = 1
x_dim = 1
sig1 = 1
sig2 = 0.1
w_mix = 0.5

# prior
prior = D.Uniform(low=-10*torch.ones(theta_dim), high=10*torch.ones(theta_dim))

# simulator
def simulator(mus):
    n = mus.shape[0]
    n1 = int(w_mix * n)
    n2 = n - n1
    x1 = D.Normal(torch.zeros(theta_dim), torch.tensor(sig1)).sample((n1,))
    x2 = D.Normal(torch.zeros(theta_dim), torch.tensor(sig2)).sample((n2,))
    x = torch.cat((x1,x2))
    return mus + x

# surrogate prior
sur_prior = sutils.BoxUniform(low=-10*torch.ones(x_dim), high=10*torch.ones(x_dim)) 

# different budgets tested
samples_len = [100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000]

# x_obs
x_obs = torch.zeros(1, x_dim)

# true_sample
mix = D.Categorical(torch.tensor([w_mix, 1-w_mix]))
comp = D.Normal(torch.zeros(2,), torch.tensor((sig1,sig2)))
gmm = D.MixtureSameFamily(mix, comp)
true_sample = gmm.sample((5000,1))

