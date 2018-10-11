import numpy as np

class RunningStatistic(object):
    def __init__(self, phis, aggregator):
        self.phis = phis
        self.phisums = [0] * len(phis)
        self.nphis = len(phis)
        self.aggregator = aggregator
    
    def observe(self, *x):
        for i in range(self.nphis):
            self.phisums[i] += self.phis[i](*x)
    
    def aggregate(self):
        return self.aggregator(*self.phisums)


class RunningMean(RunningStatistic):
    def __init__(self):
        super(RunningMean, self).__init__(phis=[lambda x: x, lambda x: 1],
                                          aggregator=lambda sumx, count: float(sumx)/count)


class RunningSTD(RunningStatistic):
    def __init__(self):
        super(RunningSTD, self).__init__(phis=[lambda x: x,lambda x: x*x, lambda x: 1],
                                          aggregator=lambda sumx, sumx2, count: np.sqrt(float(sumx2)/count-(float(sumx)/count)**2) )

class ElementWiseRunningStatistic():
    def __init__(self, rs):
        self.rs = rs
        self.shape = None

    def observe(self, *x):
        if self.shape is None:
            self.shape = x[0].shape
            self.rs_arrays = np.array([[self.rs() for j in range(self.shape[1])] for i in range(self.shape[0])])
            self.ij = [(i,j) for j in range(self.shape[1]) for i in range(self.shape[0])]
        for i, j in self.ij:
            self.rs_arrays[i,j].observe(*[xx[i,j] for xx in x])

    def aggregate(self):
        return np.array([[self.rs_arrays[i,j].aggregate() for j in range(self.shape[1])] for i in range(self.shape[0])])

def main():
    from functools import reduce
    import operator
    shape = (5, 1, 10)
    prodshape = reduce(operator.mul, shape)

    x = np.random.randn(prodshape).reshape(shape)
    y = np.random.randn(prodshape).reshape(shape)

    mrm = ElementWiseRunningStatistic(RunningMean)
    for xx in x:
        mrm.observe(xx)
    assert np.allclose(mrm.aggregate(), np.mean(x, axis=0))

    rstd = ElementWiseRunningStatistic(RunningSTD)
    for xx in x:
        rstd.observe(xx)
        print(xx)
    assert np.allclose(rstd.aggregate(), np.std(x, axis=0))
    print(rstd.aggregate(),np.std(x,axis=0))
    
    # mrrho = ElementWiseRunningStatistic(RunningRho)
    # for xx, yy in zip(x, y):
    #     mrrho.observe(xx, yy)
    # agged = mrrho.aggregate()
    #
    # for i, j in itertools.product(*list(map(range, xx.shape))):
    #     rhoij = rho(x[:, i, j].flatten(), y[:, i, j].flatten())
    #     assert np.allclose(agged[i, j], rhoij)

if __name__=="__main__":
    main()