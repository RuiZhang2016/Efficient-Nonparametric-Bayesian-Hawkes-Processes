import time, logging, itertools
import numpy as np
import ceval, wh

# np.seterr(all='raise')


@wh.memodict
def cosine_indices(d_nterms):
    d, nterms = d_nterms
    return np.array(list(itertools.product(*([range(nterms)] * d)))).T.reshape((d, 1, -1))


@wh.memodict
def chipp_lambda_cosine(d_params):
    d, (a, b, p, nterms) = d_params
    m = cosine_indices(tuple([d, nterms]))
    return 1 / (a * (np.sum(m ** 2.0, axis=0) ** p) + b).flatten()


def k_cosine_nd_basis(x, params):

    a, b, p, nterms = params
    d, n = x.shape
    m = cosine_indices(tuple([d, nterms]))
    cosx = np.cos(x.reshape((d,n,1))*m)
    num = 2.0 ** (d - np.sum(m == 0, axis=0)) / (np.pi ** d)
    phi = np.sqrt(num) * np.prod(cosx, axis=0)
    return phi

# def integral_phiphi(u,d,nterms):
#     m = cosine_indices(tuple([d, nterms]))[0]
#     dm, am = m.T-m, m.T+m
#     integral = np.repeat(dm[:, :, np.newaxis]*1.0, 3, axis=2) #dm.copy()*1.0
#     non_diag_idx = np.where(~np.eye(integral.shape[0],dtype=bool))
#     integral[non_diag_idx] = np.sin(dm[non_diag_idx]*u)/2.0/dm[non_diag_idx] + np.sin(am[non_diag_idx]*u)/2.0/am[non_diag_idx]
#     integral[range(1,nterms),range(1,nterms)] = u/2.0+1.0/4.0/m[0,range(1,nterms)]*np.sin(2.0*m[0,range(1,nterms)]*u)
#     integral[0,0] = u/2.0
#     integral = integral*2.0/np.pi
#     integral[0,1:] = integral[0,1:] /np.sqrt(2)
#     integral[1:,0] = integral[1:,0] /np.sqrt(2)
#     integral = 0.5*(integral + integral.T)
#     return integral

# def integral_phiphi(u,d,nterms):
#     u = u.reshape((len(u),1,1))
#     m = cosine_indices(tuple([d, nterms]))[0]
#     dm, am = m.T-m, m.T+m
#     dm = np.repeat(dm[np.newaxis, :, :], len(u), axis=0)
#     am = np.repeat(am[np.newaxis, :, :], len(u), axis=0)
#     integral = dm.copy()*1.0
#     eye = np.eye(integral.shape[1],dtype=bool)
#     eye3d = np.repeat(eye[np.newaxis, :, :], len(u), axis=0)
#     non_diag_idx = np.where(~eye3d)
#     integral[non_diag_idx] = ((np.sin(dm*u)/2.0/dm)+np.sin(am*u)/2.0/am)[non_diag_idx]

#     integral[:, range(1,nterms), range(1,nterms)] =  (u/2.0+(1.0/4.0/m*np.sin(2.0*m*u)))[:, 0,range(1,nterms)]
#     integral[:,0,0] = u.flatten()/2.0
#     integral *= 2.0/np.pi
#     integral[:, 0,1:] /= np.sqrt(2)
#     integral[:, 1:,0] /= np.sqrt(2)
#     integral = 0.5*(integral + np.array([e.T for e in integral]))
#     return integral

def integral_phiphi(u, d, nterms):
    sqrt2 = np.sqrt(2)
    len_u = len(u)
    u = u.reshape((len_u, 1, 1))
    ud2 = u / 2.0
    m = cosine_indices(tuple([d, nterms]))[0]
    dm, am = m.T - m, m.T + m
    dm = np.repeat(dm[np.newaxis, :, :], len_u, axis=0)
    am = np.repeat(am[np.newaxis, :, :], len_u, axis=0)
    integral = dm.copy() * 1.0
    eye = np.eye(integral.shape[1], dtype=bool)
    eye3d = np.repeat(eye[np.newaxis, :, :], len_u, axis=0)
    non_diag_idx = np.where(~eye3d)
    integral[non_diag_idx] = (np.sin(dm * u)[non_diag_idx] / dm[non_diag_idx] \
                              + np.sin(am * u)[non_diag_idx] / am[non_diag_idx]) / 2.0

    integral[:, range(1, nterms), range(1, nterms)] = (ud2 + (0.25 / m * np.sin(2.0 * m * u)))[:, 0, 1:]
    integral[:, 0, 0] = ud2.flatten()
    integral *= 2.0 / np.pi
    integral[:, 0, 1:] /= sqrt2
    integral[:, 1:, 0] /= sqrt2
    integral = 0.5 * (integral + np.array([e.T for e in integral]))
    return integral
    
def k_cosine_nd(x, y, params):

    a, b, p, nterms, tilde = params

    d, nx = x.shape
    dy, ny = y.shape

    if ny > nx:
        return k_cosine_nd(y, x, params).T

    m = cosine_indices(tuple([d, nterms]))

    cosx = np.cos(x.reshape((d,nx,1))*m)
    
    num = 2.0 ** (d - np.sum(m == 0, axis=0))/ (np.pi ** d)
    const = 1.0 if tilde else 0.0

    den = (const + a * (np.sum(m ** 2.0, axis=0) ** p) + b)
    z = (num / den)

    rval = np.zeros(nx * ny, dtype=np.float32).reshape((nx, ny))
    for iy in range(ny):
        cosy = np.cos(y[:,iy].reshape((d,1,1))*m)
        rval[:,iy] = np.sum(z * np.prod(cosx * cosy, axis=0), axis=1)

    return rval


def k_cosine_nd_diag(x, params):

    a, b, p, nterms, tilde = params

    d, nx = x.shape

    m = cosine_indices(tuple([d, nterms]))

    cosx = np.cos(x.reshape((d,nx,1))*m)

    num = 2.0 ** (d - np.sum(m == 0, axis=0)) / (np.pi ** d)
    const = 1.0 if tilde else 0.0

    den = (const + a * (np.sum(m ** 2.0, axis=0) ** p) + b)
    z = (num / den)

    rval = np.sum(z*np.prod(cosx**2, axis=0), axis=1)

    return rval


# @wh.memodict
def flaxman(covfn, maxval, nu, d, eps=1e-6):
    '''
    Args:
        covfn: covfn(x,y)
        maxval: bounding box is [0,maxval]^d
        nu: number of inducing points
        d: bounding box is [0,maxval]^d
        eps: filter eigenvalues less than eps * max(abs(eigenvalue))

    Returns:

        dict(ktilde_diag=ktilde_diag, ktilde=ktilde, lambdas=lambdas, phi=phi, k=k)

    '''
    assert np.allclose(maxval, np.pi)
    xu, _ = ceval.make_grid(d, nu)
    volume = np.pi**d
    nu = xu.shape[1]
    kuu = covfn(xu,xu)
    kuu = (kuu + kuu.T) * 0.5
    evals_mat, evecs_mat = np.linalg.eig(kuu)
    M = np.real(evecs_mat / evals_mat.reshape(1,-1) * np.sqrt(nu) / np.sqrt(volume))
    lambdas = np.real((evals_mat / nu * volume).flatten())
    cutoff = eps * max(np.abs(lambdas))
    iok = lambdas > cutoff
    print('flaxman nok', sum(iok))
    lambdas = lambdas[iok]
    M = np.real(M[:,iok])
    lambdas_tilde = 1 / (1 + 1/lambdas)
    root_lambdas_tilde = np.sqrt(lambdas_tilde).reshape(1,-1)
    root_lambdas = np.sqrt(lambdas).reshape(1,-1)
    def phi(x, xu=xu, M=M, wrapped_covfn=covfn):
        k = wrapped_covfn(x,xu)
        return k @ M
    def k(x,y, root_lambdas=root_lambdas, phi=phi):
        phi_x = phi(x)
        phi_y = phi(y)
        return (phi_x * root_lambdas) @ (phi_y * root_lambdas).T
    def ktilde(x,y, root_lambdas_tilde=root_lambdas_tilde, phi=phi):
        phi_x = phi(x)
        phi_y = phi(y)
        rval = (phi_x * root_lambdas_tilde) @ (phi_y * root_lambdas_tilde).T
        return rval
    def ktilde_diag(x,root_lambdas_tilde=root_lambdas_tilde, phi=phi):
        phi_x = phi(x)
        rval = np.sum((phi_x * root_lambdas_tilde)**2, axis=1)
        return rval.flatten()

    return dict(ktilde_diag=ktilde_diag, ktilde=ktilde, lambdas=lambdas, phi=phi, k=k)


def cov_diag(x, covfn):
    x_columns = (x[:,_,None] for _ in range(x.shape[1]))
    return np.array([covfn(thisx, thisx) for thisx in x_columns]).flatten()


# @wh.memodict()
def plaincov_maker(params):
    assert params.alpha > 0.0
    return lambda x, y, gamma=params.gamma, alpha=params.alpha: gamma * wh.covexp(x, y, np.sqrt(alpha))


class GGCPCov(object):

    def __init__(self, **params):
        self.params = wh.RaisingDotDict(params)
        self.d = self.params.d
        del params
        if self.params.name == 'flaxman':
            self.plaincov = plaincov_maker(self.params)
            precomp = flaxman(self.plaincov, self.params.maxval, self.params.nu, self.params.d, self.params.eps)
            self._ktilde = precomp['ktilde']
            self._ktilde_diag = precomp['ktilde_diag']
            self._phi = precomp['phi']
            self.lambdas = precomp['lambdas']
        elif self.params.name == 'cosine':
            tilde = True
            params_tuple = (self.params.a, self.params.b, self.params.m, self.params.nterms, tilde)
            params_tuple_2 = params_tuple[:-1]
            self._ktilde = lambda x, y, params_tuple=params_tuple: k_cosine_nd(x, y, params_tuple)
            self._ktilde_diag = lambda x, params_tuple=params_tuple: k_cosine_nd_diag(x, params_tuple)
            self._phi = lambda x, params_tuple_2=params_tuple_2: k_cosine_nd_basis(x, params_tuple_2)
            self.lambdas = chipp_lambda_cosine((self.d, params_tuple_2))
            self._integral_phiphi = lambda u:integral_phiphi(u,self.d,self.params.nterms)
        else:
            raise Exception(self.params.name)
        self.v = np.sum(np.log(1.0 / (1.0 + self.lambdas)))
        self.Lambdainv = np.diag(1.0/self.lambdas)
        
    def check_x(self, x):
        d, n = x.shape
        assert d == self.d
        assert all(x.flatten()) <= np.pi
        assert all(x.flatten()) >= 0

    def ktilde(self, x, y):
        self.check_x(x)
        self.check_x(y)
        return self._ktilde(x,y)

    def ktilde_diag(self, x):
        self.check_x(x)
        return self._ktilde_diag(x)

    def phi(self, x):
        self.check_x(x)
        return self._phi(x)

    def check_orthonormal(self, ngrid=100000):
        n = int(ngrid**(1/self.d))
        x = np.linspace(0,np.pi,n)
        dx = x[1]-x[0]
        xgrid = np.array(list(itertools.product(*[x] * self.d))).T
        phi = self.phi(xgrid)
        inner = phi.T @ phi * (dx**self.d)
        inner_diag = np.diag(inner)
        inner_off_diag = ((1-np.eye(inner.shape[0])) * inner).flatten()
        wh.summary(inner_diag, rowlabel='diag')
        wh.summary(inner_off_diag, rowlabel='offdiag', no_header=True)
        wh.assert_close(inner, np.eye(inner.shape[0]), atol=0.05)

    def triangulate_ktilde_lambda_phi(self, nx=10, ny=20):
        x = np.random.rand(nx*self.d).reshape(self.d, nx) * np.pi
        y = np.random.rand(ny*self.d).reshape(self.d, ny) * np.pi
        k = self.ktilde(x,y)
        phix = self.phi(x)
        phiy = self.phi(y)
        lambdas = self.lambdas
        k2 = phix @ np.diag(1 / (1 + 1/lambdas)) @ phiy.T
        wh.assert_close(k, k2)

    def triangulate_ktilde_ktilde_diag(self, n=10):
        x = np.random.rand(n*self.d).reshape(self.d, n) * np.pi
        k = self.ktilde_diag(x)
        k2 = self.ktilde(x,x)
        wh.assert_close(np.diag(k2), k)

    def test(self):
        self.check_orthonormal()
        self.triangulate_ktilde_lambda_phi()
        self.triangulate_ktilde_ktilde_diag()

    @classmethod
    def varied_parameters(cls, params):
        assert len(set([tuple(sorted(_.keys())) for _ in params])) == 1
        ks = sorted(params[0].keys())
        ns = [len(set([_[k] for _ in params])) for k in ks]
        return [k for k, n in zip(ks, ns) if n > 1]

    @classmethod
    def parameter_values(cls, params, k):
        return [param[k] for param in params]

if __name__ == '__main__':
    m = cosine_indices(tuple([3, 32]))
    # a = np.array([[1,2,3],[4,5,6]])
    # cosx = np.cos(a.reshape((2,3,1))*m)
    # print(np.array(list(itertools.product(*([range(32)] * 2)))))