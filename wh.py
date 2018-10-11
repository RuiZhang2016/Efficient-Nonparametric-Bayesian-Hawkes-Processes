import gc, macpath, pickle, pickletools, errno, traceback, gzip, io, copy, random, bisect, os, time
from autograd.numpy.numpy_extra import ArrayNode
import numpy as np
from functools import reduce


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def expanduser(p):
    if p is None:
        return p
    else:
        return os.path.expanduser(macpath.expanduser(p))



def dump(x, filename, opener=open, optimize=False):
    gc.collect()
    filename = expanduser(filename)
    safe_mkdir(os.path.dirname(filename))
    with opener(filename, 'wb') as fp:
        if optimize:
            s = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
            s = pickletools.optimize(s)
            fp.write(s)
        else:
            pickle.dump(x, fp, pickle.HIGHEST_PROTOCOL)
    return filename


def load(filename, exts = ['', '.gz']):
    t0 = time.time()
    filename = expanduser(filename)
    tbs = list()
    for ext in exts:
        for opener in [open,gzip.open]:
            try:
                with opener(filename+ext, 'rb') as fp:
                    if opener == gzip.open:
                        with io.BufferedReader(fp) as fpb:
                            rval = pickle.load(fpb)
                    else:
                        rval = pickle.load(fp)
                gc.collect()
                t_load = time.time() - t0
                if t_load > 10:
                    print('loaded in %i seconds' % int(t_load))
                return rval
            except:
                tbs.append(traceback.format_exc())
    raise Exception('\n-------------------------------------------------\n'.join(['']+tbs+['']))


def memodict(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


class RaisingDotDict(dict):
    _raiser = object()
    def __getattr__(self, attr):
        rval = self.get(attr,RaisingDotDict._raiser)
        if id(rval) == id(RaisingDotDict._raiser):
            raise Exception(attr)
        return rval
    def copy(self):
        return RaisingDotDict(dict(self).copy())
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def sqdist(x, y=None):
    # x.shape = (d, nx)
    if y is None:
        xx = np.sum(x ** 2, axis=0).reshape(-1, 1)
        rval = -2 * np.dot(x.T, x)
        rval += xx
        rval += xx.T
        return rval
    else:
        xx = np.sum(x ** 2, axis=0).reshape(-1, 1)
        yy = np.sum(y ** 2, axis=0).reshape(1, -1)
        rval = -2 * np.dot(x.T, y)
        rval += xx
        rval += yy
        return rval


def nuggetcov(x, y, sigma):
    return sigma ** 2 * (sqdist(x, y) == 0.0)


def covexp(x, y, sigma):
    return np.exp((-0.5 / sigma ** 2) * sqdist(x, y))


def covlap(x, y, sigma):
    return np.exp((-1.0 / sigma) * np.sqrt(sqdist(x, y)))


def assert_close(x,y,z=None,verbose=False,**kw):
    if isinstance(x, ArrayNode):
        x = x.value
    if isinstance(y, ArrayNode):
        y = y.value
    lvl = np.seterr(all='warn')
    s = lambda: '\nx=\n%s\ny=\n%s\nx-y=\n%s\nx/y=%s\nz=\n%s' % (str(x), str(y), str(x-y), str(x/y), str(z))
    np.seterr(**lvl)
    if hasattr(x, 'shape') and x.shape != y.shape:
        raise Exception(str((x.shape,y.shape)))
    if not np.allclose(x,y,**kw):
        raise Exception(s())
    if verbose:
        print(s())


def numerical_grad(x, f, eps=None, df0=None):
    if eps is None:
        eps = 1e-6
    fx = f(x)
    flatten = hasattr(fx, 'flatten')
    nx = int(np.prod(x.shape))
    nf = int(np.prod(fx.shape))
    df = np.zeros((nx,nf), dtype=float) if df0 is None else df0
    x = x * 1.0
    for i in range(len(x)):
        oldxi = x[i] * 1
        x[i] = oldxi + eps
        fxi = f(x)
        x[i] = oldxi - eps
        fxm = f(x)
        x[i] = oldxi
        if flatten:
            df[i,:] = 0.5 * (fxi - fxm).flatten() / eps
        else:
            df[i,:] = 0.5 * (fxi - fxm) / eps

    return df


def numerical_hess(x, f, eps=None):
    if eps is None:
        eps = 1e-6
    g = lambda x: numerical_grad(x, f, eps=eps)
    return numerical_grad(x, g, eps=eps)


def check_grad(x, f, df, eps=1e-6, atol=1e-5, rtol=1e-3, warn=False, numdiff = numerical_grad, do_raise=True):
    df_anal = df(x)
    df_num = numdiff(x, f, eps=eps)
    df_num = df_num.reshape(df_anal.shape)
    if not np.allclose(df_anal, df_num, atol=atol, rtol=rtol):
        s = '\nanal, num, ratio, diff:\n' + '\n'.join(map(str, ([x.flatten() for x in [df_anal, df_num, df_anal/df_num, df_anal-df_num]])))
        if warn:
            print(s)
        else:
            if do_raise:
                raise Exception(s)
            else:
                return False
    return True

def covexp(x, y, sigma):
    return np.exp((-0.5 / sigma ** 2) * sqdist(x, y))



def cumsum(x):
    y = copy.deepcopy(x)
    for i in range(1, len(x)):
        y[i] = x[i] + y[i - 1]
    return y


def cumlogaddexp(x):
    y = copy.deepcopy(x)
    for i in range(1, len(x)):
        y[i] = logaddexp(x[i], y[i - 1])
    return y


def logaddexp(a,b):
    if b >= a:
        if b == -np.inf:
            return a
        else:
            return np.logaddexp(a,b)
    return np.logaddexp(a,b)



class PMF(object):
    def __init__(self, p, v=None, check=True, normalise=False, log=False):
        p = list(map(float, p))
        if normalise:
            if log:
                s = reduce(logaddexp, p)
                p = [_ - s for _ in p]
            else:
                s = sum(p)
                p = [_ / s for _ in p]
        self.p = list(p)
        self.log = log
        if self.log:
            self.cdf = cumlogaddexp(self.p)
        else:
            self.cdf = cumsum(self.p)
        self.n = len(p)
        if check:
            if self.log:
                assert np.allclose(self.cdf[-1], 0.0), self.cdf
            else:
                assert np.all(np.array(self.p) >= 0), p
                assert np.allclose(self.cdf[-1], 1.0)
        self.cdf[-1] = 99
        self.v = v if v is not None else tuple(range(self.n))
        self.vdict = {k: i for i, k in enumerate(self.v)}
        self.uniform = len(set(p)) == 1

    def __getitem__(self, v):
        return self.p[self.vdict[v]]

    def sample(self, p=False):
        if p:
            if self.n == 1:
                return self.v[0], self.p[0]
            elif self.uniform:
                return random.choice(self.v), self.p[0]
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i], self.p[i]
        else:
            if self.n == 1:
                return self.v[0]
            elif self.uniform:
                return random.choice(self.v)
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i]

    def to_string(self, indent=0, v2s=str, header=True, prefix=''):
        if header:
            s = ('\t' * indent) + prefix + super(PMF, self).__str__() + '\n'
        else:
            s = ''
        s += ('\t' * indent)
        s += ('\n' + ('\t' * indent)).join([prefix + '%.3i : %.4f = p(%s)' % (i, np.exp(p), v2s(v)) for i, (p, v) in enumerate(zip(self.p, self.v))])
        return s

    def __str__(self):
        return self.to_string()

    def disp(self, indent=0, v2s=str, header=True):
        print(self.to_string(indent=indent, v2s=v2s, header=header))

    def __len__(self):
        return len(self.p)

import gc, macpath, pickle, pickletools, errno, traceback, gzip, io, copy, random, bisect, os, time
from autograd.numpy.numpy_extra import ArrayNode
import numpy as np
from functools import reduce


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def expanduser(p):
    if p is None:
        return p
    else:
        return os.path.expanduser(macpath.expanduser(p))



def dump(x, filename, opener=open, optimize=False):
    gc.collect()
    filename = expanduser(filename)
    safe_mkdir(os.path.dirname(filename))
    with opener(filename, 'wb') as fp:
        if optimize:
            s = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
            s = pickletools.optimize(s)
            fp.write(s)
        else:
            pickle.dump(x, fp, pickle.HIGHEST_PROTOCOL)
    return filename


def load(filename, exts = ['', '.gz']):
    t0 = time.time()
    filename = expanduser(filename)
    tbs = list()
    for ext in exts:
        for opener in [gzip.open, open]:
            try:
                with opener(filename+ext, 'rb') as fp:
                    if opener == gzip.open:
                        with io.BufferedReader(fp) as fpb:
                            rval = pickle.load(fpb)
                    else:
                        rval = pickle.load(fp)
                gc.collect()
                t_load = time.time() - t0
                if t_load > 10:
                    print('loaded in %i seconds' % int(t_load))
                return rval
            except:
                tbs.append(traceback.format_exc())
    raise Exception('\n-------------------------------------------------\n'.join(['']+tbs+['']))


def memodict(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


class RaisingDotDict(dict):
    _raiser = object()
    def __getattr__(self, attr):
        rval = self.get(attr,RaisingDotDict._raiser)
        if id(rval) == id(RaisingDotDict._raiser):
            raise Exception(attr)
        return rval
    def copy(self):
        return RaisingDotDict(dict(self).copy())
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def sqdist(x, y=None):
    # x.shape = (d, nx)
    if y is None:
        xx = np.sum(x ** 2, axis=0).reshape(-1, 1)
        rval = -2 * np.dot(x.T, x)
        rval += xx
        rval += xx.T
        return rval
    else:
        xx = np.sum(x ** 2, axis=0).reshape(-1, 1)
        yy = np.sum(y ** 2, axis=0).reshape(1, -1)
        rval = -2 * np.dot(x.T, y)
        rval += xx
        rval += yy
        return rval


def nuggetcov(x, y, sigma):
    return sigma ** 2 * (sqdist(x, y) == 0.0)


def covexp(x, y, sigma):
    return np.exp((-0.5 / sigma ** 2) * sqdist(x, y))


def covlap(x, y, sigma):
    return np.exp((-1.0 / sigma) * np.sqrt(sqdist(x, y)))


def assert_close(x,y,z=None,verbose=False,**kw):
    if isinstance(x, ArrayNode):
        x = x.value
    if isinstance(y, ArrayNode):
        y = y.value
    lvl = np.seterr(all='warn')
    s = lambda: '\nx=\n%s\ny=\n%s\nx-y=\n%s\nx/y=%s\nz=\n%s' % (str(x), str(y), str(x-y), str(x/y), str(z))
    np.seterr(**lvl)
    if hasattr(x, 'shape') and x.shape != y.shape:
        raise Exception(str((x.shape,y.shape)))
    if not np.allclose(x,y,**kw):
        raise Exception(s())
    if verbose:
        print(s())


def numerical_grad(x, f, eps=None, df0=None):
    if eps is None:
        eps = 1e-6
    fx = f(x)
    flatten = hasattr(fx, 'flatten')
    nx = int(np.prod(x.shape))
    nf = int(np.prod(fx.shape))
    df = np.zeros((nx,nf), dtype=float) if df0 is None else df0
    x = x * 1.0
    for i in range(len(x)):
        oldxi = x[i] * 1
        x[i] = oldxi + eps
        fxi = f(x)
        x[i] = oldxi - eps
        fxm = f(x)
        x[i] = oldxi
        if flatten:
            df[i,:] = 0.5 * (fxi - fxm).flatten() / eps
        else:
            df[i,:] = 0.5 * (fxi - fxm) / eps

    return df


def numerical_hess(x, f, eps=None):
    if eps is None:
        eps = 1e-6
    g = lambda x: numerical_grad(x, f, eps=eps)
    return numerical_grad(x, g, eps=eps)


def check_grad(x, f, df, eps=1e-6, atol=1e-5, rtol=1e-3, warn=False, numdiff = numerical_grad, do_raise=True):
    df_anal = df(x)
    df_num = numdiff(x, f, eps=eps)
    df_num = df_num.reshape(df_anal.shape)
    if not np.allclose(df_anal, df_num, atol=atol, rtol=rtol):
        s = '\nanal, num, ratio, diff:\n' + '\n'.join(map(str, ([x.flatten() for x in [df_anal, df_num, df_anal/df_num, df_anal-df_num]])))
        if warn:
            print(s)
        else:
            if do_raise:
                raise Exception(s)
            else:
                return False
    return True

def covexp(x, y, sigma):
    return np.exp((-0.5 / sigma ** 2) * sqdist(x, y))



def cumsum(x):
    y = copy.deepcopy(x)
    for i in range(1, len(x)):
        y[i] = x[i] + y[i - 1]
    return y


def cumlogaddexp(x):
    y = copy.deepcopy(x)
    for i in range(1, len(x)):
        y[i] = logaddexp(x[i], y[i - 1])
    return y


def logaddexp(a,b):
    if b >= a:
        if b == -np.inf:
            return a
        else:
            return np.logaddexp(a,b)
    return np.logaddexp(a,b)



class PMF(object):
    def __init__(self, p, v=None, check=True, normalise=False, log=False):
        p = list(map(float, p))
        if normalise:
            if log:
                s = reduce(logaddexp, p)
                p = [_ - s for _ in p]
            else:
                s = sum(p)
                p = [_ / s for _ in p]
        self.p = list(p)
        self.log = log
        if self.log:
            self.cdf = cumlogaddexp(self.p)
        else:
            self.cdf = cumsum(self.p)
        self.n = len(p)
        if check:
            if self.log:
                assert np.allclose(self.cdf[-1], 0.0), self.cdf
            else:
                assert np.all(np.array(self.p) >= 0), p
                assert np.allclose(self.cdf[-1], 1.0)
        self.cdf[-1] = 99
        self.v = v if v is not None else tuple(range(self.n))
        self.vdict = {k: i for i, k in enumerate(self.v)}
        self.uniform = len(set(p)) == 1

    def __getitem__(self, v):
        return self.p[self.vdict[v]]

    def sample(self, p=False):
        if p:
            if self.n == 1:
                return self.v[0], self.p[0]
            elif self.uniform:
                return random.choice(self.v), self.p[0]
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i], self.p[i]
        else:
            if self.n == 1:
                return self.v[0]
            elif self.uniform:
                return random.choice(self.v)
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i]

    def to_string(self, indent=0, v2s=str, header=True, prefix=''):
        if header:
            s = ('\t' * indent) + prefix + super(PMF, self).__str__() + '\n'
        else:
            s = ''
        s += ('\t' * indent)
        s += ('\n' + ('\t' * indent)).join([prefix + '%.3i : %.4f = p(%s)' % (i, np.exp(p), v2s(v)) for i, (p, v) in enumerate(zip(self.p, self.v))])
        return s

    def __str__(self):
        return self.to_string()

    def disp(self, indent=0, v2s=str, header=True):
        print(self.to_string(indent=indent, v2s=v2s, header=header))

    def __len__(self):
        return len(self.p)


def sample_degenerate_mvn(mu, sigma, n, seed=None, small=1e-6):

    if mu is None:
        mu = np.zeros(sigma.shape[0])
    mu = mu.reshape(-1, 1)
    if n == 0:
        return np.array([]).reshape((0,sigma.shape[1]))
    d = sigma.shape[0]
    assert sigma.shape[1] == d
    assert len(mu) == d

    U, S, V = np.linalg.svd(0.5*(sigma+sigma.T))
    S[S<(small*max(np.abs(S)))]=0
    m = sum(S!=0)

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    n01 = np.random.randn(m * n).reshape((m, n))
    n01 = np.vstack((n01,np.zeros(((d-m)*n)).reshape((d-m, n))))
    rval = np.dot(np.dot(U, np.diag(np.sqrt(S))), n01) + mu

    if seed is not None:
        np.random.set_state(state)

    return rval
