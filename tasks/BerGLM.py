import torch
import torch.distributions as D # for distributions
from sbi import utils as sutils
import numpy as np

# hyperparams
x_dim = 10
theta_dim = 10

# prior
M = theta_dim - 1
Dd = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
F = Dd @ Dd + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
Binv = torch.zeros(size=(M + 1, M + 1))
Binv[0, 0] = 0.5  # offset
Binv[1:, 1:] = torch.matmul(F.T, F)  # filter
prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}
prior = D.MultivariateNormal(**prior_params)
prior.set_default_validate_args(False)

stimulus_I = torch.load("./saved/stimulus_I.pt")
design_matrix = torch.load("./saved/design_matrix.pt") # V in paper

# simulator
def simulator(parameters: torch.Tensor, return_both: bool = False) -> torch.Tensor:
    """Simulates model for given parameters
    If `return_both` is True, will additionally return spike train not reduced to summary features
    """
    data = []
    data_raw = []
    for b in range(parameters.shape[0]):
      # Simulate GLM
      psi = torch.matmul(design_matrix, parameters[b, :])
      z = 1 / (1 + torch.exp(-psi))
      y = (torch.rand(design_matrix.shape[0]) < z).float()

      # Calculate summary statistics
      num_spikes = torch.sum(y).unsqueeze(0)
      sta = torch.nn.functional.conv1d(y.reshape(1, 1, -1), stimulus_I.reshape(1, 1, -1), padding=8).squeeze()[-9:]
      data.append(torch.cat((num_spikes, sta)))

      if return_both:
          data_raw.append(y)

    if not return_both:
        return torch.stack(data)
    else:
        return torch.stack(data), torch.stack(data_raw)

# surrogate prior 
low = torch.tensor([0, -35, -35, -35, -35, -35, -35, -35, -35, -35])
high = torch.tensor([100, 25, 25, 25, 25, 25, 25, 25, 25, 25])
sur_prior = sutils.BoxUniform(low=low, high=high)

# different budgets tested
samples_len = [250, 500, 1000, 1500, 2500, 5000, 7500, 10000]

# x_obs
# see https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/bernoulli_glm/files/
# num_observation_2
theta_true = torch.tensor([1.4452896,-1.5240457,-2.6627376,-2.057486,-1.0255841,-0.33056775,0.4173284,-0.39998925,0.09306164,0.39093223])
x_obs_raw = torch.tensor([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                          1.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,
                          1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0])
x_obs = design_matrix.T @ x_obs_raw

# true_sample
ts = np.loadtxt("./saved/BerGLM_reference_posterior_samples.csv", delimiter=',', skiprows=1)
true_sample = torch.tensor(ts, dtype=torch.float32)


