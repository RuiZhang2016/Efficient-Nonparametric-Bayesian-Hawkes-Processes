import logging
import numpy as np
import matplotlib.pyplot as plt


def gpplot(x, mu, lower=None, upper=None, delta=None, edgecol='k', fillcol='gray', alpha=0.3, mulabel=None, filllabel=None, **kwargs):
    arr = lambda x: x if isinstance(x, np.ndarray) else np.array(x).flatten()
    mu, x, = list(map(arr, (mu, x)))
    mu = mu.flatten()
    x = x.flatten()

    if delta is not None:
        assert lower is None and upper is None
        delta = arr(delta)
        lower = mu - delta
        upper = mu + delta
    lower, upper = list(map(arr, (lower, upper)))
    lower = lower.flatten()
    ind = ~(np.isnan(x) | np.isnan(mu) | np.isnan(lower) | np.isnan(upper))
    if any(~ind):
        logging.warning('%i / %i nans in gpplot' % (sum(~ind), len(ind)))
    x, mu, lower, upper = [_[ind] for _ in (x, mu, lower, upper)]
    if np.any(lower > mu):
        logging.warning('lower > mu in gpplot')
    upper = upper.flatten()
    if np.any(upper < mu):
        logging.warning('upper < mu in gpplot')

    plots = []

    plots.append(plt.plot(x, mu, color=edgecol, label=mulabel))

    kwargs['linewidth'] = 0.5
    if not 'alpha' in list(kwargs.keys()):
        kwargs['alpha'] = alpha
    plots.append(plt.gca().fill(np.hstack((x, x[::-1])), np.hstack((upper, lower[::-1])), color=fillcol, **kwargs))

    # this is the edge:
    plots.append(plt.plot(x, upper, color=edgecol, linewidth=0.2))
    plots.append(plt.plot(x, lower, color=edgecol, linewidth=0.2, label=filllabel))

    return plots
