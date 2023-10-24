import torch

def MMD2(x,y):
    # squared (2-norm) distances
    n_x, n_y = x.shape[0], y.shape[0]
    dxx = torch.pdist(x)**2
    dyy = torch.pdist(y)**2
    dxy = (torch.cdist(x,y)**2).reshape(-1)
    # median heuristic
    scale = torch.median(torch.sqrt(torch.cat([dxx, dxy, dyy])))
    # RBF kernel
    c = -0.5 / (scale ** 2)
    XX = torch.exp(dxx*c)
    YY = torch.exp(dyy*c)
    XY = torch.exp(dxy*c)
    return XX.sum()/(n_x*(n_x-1)/2) + YY.sum()/(n_y*(n_y-1)/2) - 2. * XY.sum()/(n_x*n_y)

def energy_dist(X,Y):
    dxx = torch.cdist(X,X).reshape(-1)
    dyy = torch.cdist(Y,Y).reshape(-1)
    dxy = torch.cdist(X,Y).reshape(-1)
    nrg = 2 * dxy.mean(dim=0) - dxx.mean(dim=0) - dyy.mean(dim=0)
    return nrg

def HH_metrics(sample, true):
    q1 = torch.tensor([0.15, 0.85], dtype=sample.dtype) # quantiles 
    # q2 = torch.tensor([0.5], dtype=torch.float64) # median
    iqr = torch.quantile(sample, q1, dim=0) # inter quantile range
    median = torch.median(sample, dim=0).values # torch.quantile(sample, q2, dim=0) # median
    mean = torch.mean(sample, dim=0)
    sds = torch.sqrt(torch.var(sample, dim=0))

    m1 = torch.linalg.norm(median-true)  # how close is the true parameter to the median?
    m2 = torch.linalg.norm(mean-true) # how close is the true parameter to the median?
    m3 = torch.mean(abs(iqr[0,:]-iqr[1,:])) # how big is the IQR?
    m4 = torch.mean(sds) # how big is the variance
    
    return torch.tensor([m1, m2, m3, m4])