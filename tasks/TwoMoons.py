import torch
import numpy as np
import torch.distributions as D # for distributions
from sbi import utils as sutils

# hyperparams
theta_dim = 2
x_dim = 2
n = 10000
a_l = -np.pi/2
a_u = np.pi/2
r_mu = 0.1
r_sig = 0.01

# prior
prior = sutils.BoxUniform(low=-1*torch.ones(theta_dim), high=1*torch.ones(theta_dim)) 

# simulator
def simulator(theta):
    n = theta.shape[0] 
    a = D.Uniform(torch.tensor(a_l), torch.tensor(a_u)).sample((n,))
    r = D.Normal(torch.tensor(r_mu), torch.tensor(r_sig)).sample((n,))
    px = r*torch.cos(a)+0.25
    py = r*torch.sin(a)
    x = px - torch.abs(torch.sum(theta, dim=1))/np.sqrt(2)
    x = x.unsqueeze(1)
    y = py + (theta[:,1]-theta[:,0])/np.sqrt(2)
    y = y.unsqueeze(1)
    return torch.cat((x,y), 1)
    
# surrogate prior (took rough boundaries of x's generated from prior theta's) 
x_min = torch.tensor([-1.5,-2]) 
x_max = torch.tensor([0.6, 2])
sur_prior = sutils.BoxUniform(low=x_min, high=x_max) 

# different budgets tested
samples_len = [100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000]

# x_obs
x_obs = torch.zeros(1, x_dim)

# true_sample
def gen_posterior_samples(obs=torch.tensor([0.0, 0.0]), prior=None, n_samples=1):
    # use opposite rotation as above
    c = 1/np.sqrt(2)

    theta = torch.zeros((n_samples, 2))
    for i in range(n_samples):
        p = simulator(torch.zeros(1,2))
        q = torch.zeros(2)
        q[0] = p[0,0] - obs[0]
        q[1] = obs[1] - p[0,1]

        if np.random.rand() < 0.5:
            q[0] = -q[0]

        theta[i, 0] = c * ( q[0] - q[1] )
        theta[i, 1] = c * ( q[0] + q[1] )

    return theta
    
true_sample = gen_posterior_samples(n_samples=10000)

