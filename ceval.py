import itertools
import numpy as np
import wh

def toy_data(n,ngrid,omega):
    x_grid = np.linspace(0,np.pi,ngrid)
    dx = x_grid[1] - x_grid[0]
    p = np.sin(x_grid * omega) + 1
    p = p / np.sum(p)
    intensity = p * n / dx
    x_grid = x_grid.reshape(1,-1)
    x = sample_pp(intensity, x_grid, n_is_mu=False)
    return x, intensity, x_grid


def sample_pp(lambda_grid, x_grid, n_is_mu = True, force_n=None): 
    """Simulation according to lambda"""
    d = x_grid.shape[0]
    xu = sorted(set(x_grid[0,:]))
    dx = xu[1] - xu[0]
    volume = dx ** d
    mu = np.sum(lambda_grid * volume)
    if force_n is not None:
        n = force_n
    else:
        n = int(mu) if n_is_mu else np.random.poisson(mu)
    p = lambda_grid / np.sum(lambda_grid)
    pmf = wh.PMF(p=p,v=tuple(map(tuple, x_grid.T)))
    x = np.array([pmf.sample() for _ in range(n)]).T
    x = x.reshape(d, -1)
    x += (np.random.rand(n).reshape(x.shape) - 0.5) * dx
    x[x<min(x_grid.flatten())] = min(x_grid.flatten())
    x[x>max(x_grid.flatten())] = max(x_grid.flatten())
    return x

def expected_log_p(lam, gam, measure):
    # E_{x~PP(lambda)} log(p(x|PP(gamma)))
    # derivation in the paper:
    return (sum(lam * np.log(gam)) - sum(gam)) * measure

def make_grid(d, n, max_grid):
    grid = np.linspace(1e-6,max_grid-1e-6,n) #
    
    measure = (grid[1] - grid[0]) ** d
    z = np.array(list(itertools.product(*[grid] * d))).T
    assert z.shape[0] == d, (d, z.shape)
    return z, measure

def generate_z(d,n,max_grid):
    #grid = np.logspace(-8,np.log(max_grid)-1e-4,n,base=np.e)
    grid = np.linspace(1e-6,max_grid-1e-6,n)
    z = np.array(list(itertools.product(*[grid] * d))).T
    assert z.shape[0] == d, (d, z.shape)
    return z

def oned_testfunc(func_seed):
    np.random.seed(func_seed)
    xgrid, measuregrid = make_grid(1, 1024)
    k = 5**2*wh.covexp(xgrid, xgrid, 0.5)
    k += np.eye(k.shape[0]) * 1e-6
    f = wh.sample_mvn(None, k, 1, func_seed)
    intensity = 0.5 * f.flatten()**2
    return dict(x=xgrid, intensity=intensity, measure=measuregrid, name='synthetic_%i' % func_seed)
