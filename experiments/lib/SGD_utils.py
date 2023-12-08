import numpy as np

class SGD:
    """
    Exact SGD dynamics
    """
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, nsamp, batchsize, seed):
        """
        lr: learning rate
        q: model
        grad_q: gradient of the model
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.std_epsilon = std_epsilon
        # uncorrelated X and Y data
        self.x = self.state.normal(size=nsamp)
        self.y = self.state.normal(size=nsamp, scale=std_epsilon)
        
    def update(self, w_old):
        xb = self.state.choice(self.x, self.nb, replace=False)
        yb = self.state.choice(self.y, self.nb, replace=False)
        
        xi_xx = np.mean(xb*xb)
        xi_xy = np.mean(xb*yb)
        return w_old - self.lr*(xi_xx * self.q(w_old) - xi_xy) * self.grad_q(w_old)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)

class SGDInfiniteData:
    """
    Exact SGD dynamics
    """
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, batchsize, seed):
        """
        lr: learning rate
        q: model
        grad_q: gradient of the model
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.std_epsilon = std_epsilon
        
    def update(self, w_old):
        xb = self.state.normal(size=self.nb)
        yb = self.state.normal(size=self.nb, scale=self.std_epsilon)
        
        xi_xx = np.mean(xb*xb)
        xi_xy = np.mean(xb*yb)
        return w_old - self.lr*(xi_xx * self.q(w_old) - xi_xy) * self.grad_q(w_old)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)

def save_data(std_epsilon, lr, q, grad_q, w_init, nsamp, batchsize, seed, nsteps, filename):
    sgd = SGD(std_epsilon, lr, q, grad_q, w_init, nsamp, batchsize, seed)
    sgd.evolve(nsteps)
    np.savetxt(filename + f'stde{std_epsilon}_lr{lr}_winit_{w_init}_nsamp{nsamp}_b{batchsize}_seed{seed}_nsteps{nsteps}.dat', sgd.w)

class ApproxSGD:
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, batchsize, seed):
        """
        lr: learning rate
        q: model
        grad_q: gradient of the model
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.ve = std_epsilon**2.
        # variance of x data (set to 1 by default)
        self.v = 1.
        
    def update(self, w_old):
        xi1, xi2 = self.state.normal(size=2, scale=np.sqrt(2))

        q = self.q(w_old)

        # noise-independent term
        drift = self.v*q
        # noise term
        diffusion = (self.v*q*xi1 - np.sqrt(self.v*self.ve/2.)*xi2)/np.sqrt(self.nb)
        # common factor in front of every term
        factor = self.lr*self.grad_q(w_old)
        
        return w_old - factor*(drift + diffusion)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)

class TrimSGD:
    """
    GD + noise with non-Gaussian tails
    """
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, batchsize, seed, threshold):
        """
        lr: learning rate
        q: model
        grad_q: gradient of the model
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.ve = std_epsilon**2.
        # variance of x data (set to 1 by default)
        self.v = 1.
        self.threshold = threshold
        
    def update(self, w_old):
        xi1, xi2 = self.state.normal(size=2, scale=np.sqrt(2))

        # reject samples above the threshold
        while np.abs(xi2) > self.threshold:
            xi2 = self.state.normal(scale=np.sqrt(2))
            
        q = self.q(w_old)

        # noise-independent term
        drift = self.v*q
        # noise term
        diffusion = (self.v*q*xi1 - np.sqrt(self.v*self.ve/2.)*xi2)/np.sqrt(self.nb)
        # common factor in front of every term
        factor = self.lr*self.grad_q(w_old)
        
        return w_old - factor*(drift + diffusion)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)

class TailsSGD:
    """
    GD + noise with non-Gaussian tails
    """
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, batchsize, seed, t):
        """
        lr: learning rate
        q: model
        grad_q: gradient of the model
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.ve = std_epsilon**2.
        # variance of x data (set to 1 by default)
        self.v = 1.
        self.t = t
        
    def update(self, w_old):
        # degrees of freedom for Student's
        df = 2.4
        # variance of Student
        var = df/(df-2.)
        # adjusted scale such that variance is 2
        scale = np.sqrt(2-var*self.t**2)/(1-self.t)
        xi1, xi2 = (1-self.t)*self.state.normal(size=2, scale=scale) + self.t*self.state.standard_t(df=df, size=2)
            
        q = self.q(w_old)

        # noise-independent term
        drift = self.v*q
        # noise term
        diffusion = (self.v*q*xi1 - np.sqrt(self.v*self.ve/2.)*xi2)/np.sqrt(self.nb)
        # common factor in front of every term
        factor = self.lr*self.grad_q(w_old)
        
        return w_old - factor*(drift + diffusion)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)
