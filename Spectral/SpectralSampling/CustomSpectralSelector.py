from apricot import BaseSelection, BaseGraphSelection
import numpy
from apricot.optimizers import LazyGreedy
from apricot.optimizers import ApproximateLazyGreedy
from apricot.optimizers import SieveGreedy

from tqdm import tqdm

from numba import njit
from numba import prange

class CustomSpectralSelector(BaseSelection):
    '''This is a minimal implementation of a feature-based function with a sqrt.'''
    
    def __init__(self, n_samples, initial_subset=None, optimizer='two-stage', 
        optimizer_kwds={}, n_jobs=1, random_state=None, 
        verbose=False):
        
        super(CustomSpectralSelector, self).__init__(n_samples=n_samples, 
            initial_subset=initial_subset, optimizer=optimizer, 
            optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, random_state=random_state, 
            verbose=verbose)

    def _initialize(self, X):
        # The cached values will be the column sums.
        self.current_values = numpy.zeros(X.shape[1])
        
        # We should also keep track of the total gain thus far
        # so we can calculate the marginal gain of adding each
        # element quickly.
        self.total_gain = 0.0
        super(CustomSpectralSelector, self)._initialize(X)

    def _calculate_gains(self, X, idxs=None):
        # The gains are the increase in the objective. This can be
        # calculated as the objective value of each example minus
        # the stored accumulated gain. Given that this is trivially
        # vectorizable, the code is not actually complex.

        idxs = idxs if idxs is not None else self.idxs
        gains = numpy.sqrt(X[idxs] + self.current_values).sum(axis=1) - self.total_gain
        return gains

    def _select_next(self, X, gain, idx):
        # Because we are storing column sums we only need to do an
        # element-wise addition to update the cached values and
        # another addition to store the accumulated gain.
        
        self.current_values += X
        self.total_gain += gain
        super(CustomSpectralSelector, self)._select_next(X, gain, idx)

        
if __name__ == '__main__':  
    import time
    
    X = numpy.exp(numpy.random.randn(1000, 100))
    model = CustomSpectralSelector(3, optimizer='approximate-lazy')
    start = time.time()
    model.fit(X)
    print("Time:", time.time()-start)
    
    
    