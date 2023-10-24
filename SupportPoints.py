import torch

def ccp(x, Y):
    x_new = torch.zeros_like(x)
    n = x.shape[0]
    N = Y.shape[0]
    for i in range(n):
        xi = x[i,:]
        ti = (xi-Y).pow(2)
        ti2 = torch.sum(ti, dim=1).sqrt()
        ti3 = ti2[ti2 != 0]
        qi = torch.sum(ti3.pow(-1))
        
        ui = xi-x
        ui2 = torch.sum((xi-x).pow(2), dim=1).sqrt()
        a = ui[ui2!=0,:]
        b = ui2[ui2!=0].unsqueeze(1)
        term1 = torch.sum(a/b, dim=0)
        
        a = Y[ti2!=0,:]
        
        b = ti2[ti2!=0].unsqueeze(1)
        term2 = torch.sum(a/b, dim=0)
        
        x_new[i,:] = qi.pow(-1) * ((N/n) *term1 + term2)
    return x_new

def do_ccp(Y, n=100, n_rounds_max=250, tol=1e-3): 
    N = Y.shape[0]
    p = torch.tensor([1/N]*N)
    idx = p.multinomial(num_samples=n, replacement=False)
    x = Y[idx, :]
    for j in range(n_rounds_max):
        x2 = ccp(x,Y)
        d = torch.sum((x-x2).pow(2))
        if (d < tol):
            return x2, j
        x = x2
    return x, j

def constrain_points(x, dist):
  keep = torch.isfinite(dist.log_prob(x))
  return x[keep]