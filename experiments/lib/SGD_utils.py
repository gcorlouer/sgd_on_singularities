import numpy as np

class SGD:
    """
    Exact SGD dynamics
    """
    def __init__(self, lr, q, grad_q, w_init, nsamp, batchsize, seed):
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
        # uncorrelated X and Y data
        self.x, self.y = self.state.normal(size=(2, nsamp))
        
    def update(self, w_old):
        xb = self.state.choice(self.x, self.nb)
        yb = self.state.choice(self.y, self.nb)
        
        xi_xx = np.mean(xb*xb)
        xi_xy = np.mean(xb*yb)
        return w_old - self.lr*(xi_xx * self.q(w_old) - xi_xy) * self.grad_q(w_old)
    
    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)

def save_data(lr, q, grad_q, w_init, nsamp, batchsize, seed, nsteps, filename):
    sgd = SGD(lr, q, grad_q, w_init, nsamp, batchsize, seed)
    sgd.evolve(nsteps)
    np.savetxt(filename + f'lr{lr}_winit_{w_init}_nsamp{nsamp}_b{batchsize}_seed{seed}_nsteps{nsteps}.dat', sgd.w)