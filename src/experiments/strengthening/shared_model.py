from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'authority_calibration'))
from screen import torch, features as torch_features, derivative as torch_derivative, K

DT = .002
R = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
Q = np.diag([1e-12, 1e-12, 1e-7, 1e-7])
I4 = np.eye(4)


def terms(x, u):
    q1, q2, v1, v2 = np.moveaxis(x, -1, 0)
    kappa = .6255677234174437*.2*.25569305436052964
    d = .05086984812807257
    a = .031887199591513114+d+.6255677234174437*.2**2+2*kappa*np.cos(q2)
    b = d+kappa*np.cos(q2); det = a*d-b*b
    ga = (.5234602302310271*.2+.6255677234174437*.2)*9.81
    gb = .6255677234174437*.25569305436052964*9.81
    f0 = -ga*np.sin(q1)-gb*np.sin(q1+q2)+kappa*np.sin(q2)*(2*v1*v2+v2*v2)
    f1 = -gb*np.sin(q1+q2)-kappa*np.sin(q2)*v1*v1
    base = np.stack([v1, v2, (d*f0-b*f1)/det, (-b*f0+a*f1)/det], -1)
    gain = np.stack([-b*u/det, a*u/det], -1)
    drag1 = np.stack([-d*v1*.03/det, b*v1*.03/det], -1)
    drag2 = np.stack([b*v2*.03/det, -a*v2*.03/det], -1)
    return base, np.stack([gain, drag1, drag2], -1)


def rhs(x, u, theta):
    f, phi = terms(x, u)
    f[..., 2:] += np.einsum('...ij,...j->...i', phi, theta)
    return f


def step(x, u, theta, dt=DT):
    a = rhs(x, u, theta); b = rhs(x+dt*a/2, u, theta)
    c = rhs(x+dt*b/2, u, theta); d = rhs(x+dt*c, u, theta)
    return x+dt*(a+2*b+2*c+d)/6


class Calibration:
    def __init__(self, y, adaptive=True):
        self.theta = np.array([1., 0., 0.]); self.xh = y.copy(); self.y = y.copy()
        self.cov = R.copy(); self.G = np.eye(3)*1e-4; self.r = self.G@self.theta
        self.v0 = y[2:].copy(); self.phi = np.zeros((2, 3)); self.f = np.zeros(2)
        self.adaptive = adaptive

    def update(self, u, yn, k):
        f0, p0 = terms(self.y, u); f1, p1 = terms(yn, u)
        self.phi += (p0+p1)*.001; self.f += (f0[2:]+f1[2:])*.001
        perturb = I4*1e-5
        jac = ((rhs(self.xh[None]+perturb, u, self.theta)
                -rhs(self.xh[None]-perturb, u, self.theta))/2e-5).T
        F = I4+DT*jac; cov = F@self.cov@F.T+Q
        gain = np.linalg.solve(cov+R, cov.T).T
        pred = step(self.xh, u, self.theta)
        self.xh = pred+gain@(yn-pred)
        J = I4-gain; self.cov = J@cov@J.T+gain@R@gain.T
        self.y = yn
        if (k+1) % 100 == 0:
            if self.adaptive:
                target = yn[2:]-self.v0-self.f
                self.G = .95*self.G+self.phi.T@self.phi
                self.r = .95*self.r+self.phi.T@target
                self.theta = np.clip(np.linalg.solve(self.G, self.r),
                                     np.array([.4, 0, 0], dtype=np.float32).astype(float),
                                     np.array([1.6, 2, 2], dtype=np.float32).astype(float))
            self.v0 = yn[2:].copy(); self.phi.fill(0); self.f.fill(0)


def noise_stream(seed=51, n=64):
    gen = torch.Generator().manual_seed(seed)
    sigma = torch.tensor([.001, .001, .01, .01], dtype=torch.float64)
    return np.stack([(torch.randn((n, 4), generator=gen, dtype=torch.float64)*sigma).numpy()
                     for _ in range(10001)])


def validate():
    rng = np.random.default_rng(9813)
    x = rng.uniform([-6, -6, -12, -12], [6, 6, 12, 12], (128, 4))
    u = rng.uniform(-.5, .5, 128)
    theta = rng.uniform([.4, 0, 0], [1.6, 2, 2], (128, 3))
    tx = torch.tensor(x); tu = torch.tensor(u); tt = torch.tensor(theta)
    feature_error = float(abs(terms(x, u)[1]-torch_features(tx, tu).numpy()).max())
    dynamics_error = float(abs(rhs(x, u, theta)-torch_derivative(tx, tu, tt).numpy()).max())
    # Independently evaluate one EKF prediction/correction with nonnominal gains
    # and drag, rather than checking only the nominal observer.
    from screen import integrate as torch_integrate
    yn = step(x, u, theta)+rng.normal(size=x.shape)*[.001,.001,.01,.01]
    eye=torch.eye(4,dtype=torch.float64);delta=eye[None]*1e-5
    jac=((torch_derivative(tx[:,None]+delta,tu[:,None],tt[:,None])
          -torch_derivative(tx[:,None]-delta,tu[:,None],tt[:,None]))/2e-5).transpose(-1,-2)
    F=eye+DT*jac;pc=torch.tensor(R).repeat(len(x),1,1)
    pc=F@pc@F.transpose(-1,-2)+torch.tensor(Q)
    gain=torch.linalg.solve(pc+torch.tensor(R),pc.transpose(-1,-2)).transpose(-1,-2)
    pred=torch_integrate(tx,tu,tt,DT)
    reference=pred+torch.einsum('nij,nj->ni',gain,torch.tensor(yn)-pred)
    J=eye-gain;pc=J@pc@J.transpose(-1,-2)+gain@torch.tensor(R)@gain.transpose(-1,-2)
    states=[];covariances=[]
    for i in range(len(x)):
        observer=Calibration(x[i]);observer.theta=theta[i].copy();observer.update(u[i],yn[i],0)
        states.append(observer.xh);covariances.append(observer.cov)
    observer_error=float(abs(np.asarray(states)-reference.numpy()).max())
    covariance_error=float(abs(np.asarray(covariances)-pc.numpy()).max())
    # Repeated noisy windows: compare all sufficient statistics and projections.
    cal = Calibration(x[0]); G = torch.eye(3, dtype=torch.float64)*1e-4
    r = G@torch.tensor([1., 0., 0.], dtype=torch.float64)
    ph = torch.zeros((2, 3), dtype=torch.float64); f = torch.zeros(2, dtype=torch.float64)
    y = tx[0].clone(); v0 = y[2:].clone(); errors = []
    for k in range(300):
        yn = x[(k+1) % 128]; uk = u[k % 128]
        from screen import rhs as unforced
        ph += (torch_features(y, torch.tensor(uk))+torch_features(torch.tensor(yn), torch.tensor(uk)))*.001
        f += (unforced(y, torch.tensor(0.))[2:]+unforced(torch.tensor(yn), torch.tensor(0.))[2:])*.001
        cal.update(uk, yn, k); y = torch.tensor(yn)
        if (k+1) % 100 == 0:
            G = .95*G+ph.T@ph; r = .95*r+ph.T@(y[2:]-v0-f)
            t = torch.linalg.solve(G, r).clamp(torch.tensor([.4, 0, 0]), torch.tensor([1.6, 2., 2.]))
            errors.append(float(abs(cal.theta-t.numpy()).max()))
            v0 = y[2:].clone(); ph.zero_(); f.zero_()
    assert max(feature_error, dynamics_error, observer_error, covariance_error, *errors) < 1e-8
    return dict(feature_max_abs=feature_error, dynamics_max_abs=dynamics_error,
                observer_state_max_abs=observer_error,observer_covariance_max_abs=covariance_error,
                projected_estimator_max_abs=max(errors))
