"""Two-P-joint wrist tool (SI: m, rad, s, N, N*m).

demo: python wrist_tool.py
evalaute: python wrist_tool.py --test
usage in real case: python wrist_tool.py --config geometry.json

JSON label: a, b (2x3), q_min, q_max, q_home (for each set),
rho_min, rho_max, rho_offset (for each set) (rad, m)
e.g. a=[[.04,.03,-.15],[-.04,.03,-.15]],
b=[[.04,.03,0],[-.04,.03,0]]

Model: O at center of Universal Joint, R=Ry(beta)Rx(alpha)
Cartesian Coordinate: Z --> to hand, X --> side of wrist, connector of p-joint is free to rotate
** FK use continum to deriviate, using q_(k-1) as seed with max_step_rad to prevent singularity 

rho define as p-rod length; s = stroke = rho-rho_offset
J=d(rho)/d(q); H[i,j,k]=d²rho_i/(dq_j dq_k)
qdot as (alpha_dot, beta_dot), not omega_xyz; omega_xyz=E *qdot.
tau is generalized and conjugate to q  ; +f toward hand
In case ofdynamic for F--> a, M and bias=c+g+friction is needed
rlo:= rho lower (bound)
rhi:= rho upper (bound)
rho:= \rho, the scalar of length of p-rod
"""
import argparse
import json
import unittest
import numpy as np
from scipy.optimize import least_squares


def arr(x, shape, name):
    """check and return a np array of "shape"""
    v = np.asarray(x, dtype=float)
    if v.shape != shape or not np.all(np.isfinite(v)):
        raise ValueError(f'{name}: expected finite shape {shape}')
    return v


def rotations(q):
    """return rotation matrix R R_dot R_ddot (d/dq), first derivatives (2,3,3), second derivatives (2,2,3,3)."""
    a, b = q
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    X = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    X1 = np.array([[0,0,0],[0,-sa,-ca],[0,ca,-sa]])
    X2 = np.array([[0,0,0],[0,-ca,sa],[0,-sa,-ca]])
    Y = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
    Y1 = np.array([[-sb,0,cb],[0,0,0],[-cb,0,-sb]])
    Y2 = np.array([[-cb,0,-sb],[0,0,0],[sb,0,-cb]])
    return Y@X, np.array([Y@X1,Y1@X]), np.array([[Y@X2,Y1@X1],[Y1@X1,Y2@X]])


class Wrist:
    def __init__(self, a, b, q_min, q_max, q_home,
                 rho_min=None, rho_max=None, rho_offset=None):
        self.a, self.b = arr(a,(2,3),'a'), arr(b,(2,3),'b')
        self.lo, self.hi = arr(q_min,(2,),'q_min'), arr(q_max,(2,),'q_max')
        if np.any(self.lo >= self.hi):
            raise ValueError('q_min must be less than q_max')
        self.home = self._q(q_home)
        self.rlo = arr([0,0] if rho_min is None else rho_min,(2,),'rho_min')
        self.rhi = np.full(2,np.inf) if rho_max is None else arr(rho_max,(2,),'rho_max')
        self.offset = arr([0,0] if rho_offset is None else rho_offset,(2,),'rho_offset')
        if np.any(self.rlo < 0) or np.any(self.rhi <= self.rlo):
            raise ValueError('invalid length limits')
        self.ik(self.home)

    def _q(self, q):
        """check and record s_0"""
        q = arr(q,(2,),'q')
        if np.any(q < self.lo-1e-12) or np.any(q > self.hi+1e-12):
            raise ValueError('angle outside limits')
        return q

    def _rho(self, rho):
        """check validity of rod's length"""
        rho = arr(rho,(2,),'rho')
        if np.any(rho <= 0) or np.any(rho < self.rlo) or np.any(rho > self.rhi):
            raise ValueError('length outside limits')
        return rho

    def geometry(self, q):
        """Return rho, J, H, p, u with analytic derivatives."""
        R, R1, R2 = rotations(self._q(q))
        p = self.b@R.T
        d = p-self.a
        rho = np.linalg.norm(d,axis=1)
        if np.any(rho < 1e-12):
            raise ValueError('zero-length actuator: direction undefined')
        u = d/rho[:,None]
        p1 = np.einsum('jab,ib->ija',R1,self.b) #　dp_i/dq_j, i:which rod, j:to which angle, a:to which(x,y,z), b:from which(x,y,z)
        p2 = np.einsum('jkab,ib->ijka',R2,self.b) #　ddp_i/(dq_j*dq_k), i:which rod, j:to which angle, k:to which angle, a:to which(x,y,z), b:from which(x,y,z)
        J = np.einsum('ia,ija->ij',u,p1) # s_dot = J*q_dot, dp_i/dq_j, i:which rod, j:to which angle, a:to which(x,y,z)
        H = np.empty((2,2,2))
        for i in range(2):
            H[i] = p1[i]@(np.eye(3)-np.outer(u[i],u[i]))@p1[i].T/rho[i]
            H[i] += np.einsum('a,jka->jk',u[i],p2[i])
        return rho,J,H,p,u

    def ik(self, q):
        return self._rho(self.geometry(q)[0])

    def to_stroke(self, rho):
        return self._rho(rho)-self.offset

    def from_stroke(self, stroke):
        return self._rho(arr(stroke,(2,),'stroke')+self.offset)

    def jacobian(self, q):
        return self.geometry(q)[1]

    def hessians(self, q):
        return self.geometry(q)[2]

    def diagnostics(self, q):
        s = np.linalg.svd(self.jacobian(q),compute_uv=False)
        return {'singular_values':s, 'condition':float(s[0]/s[-1]) if s[-1]>0 else np.inf}

    def _regular_J(self, q):
        """check if J^-1 available"""
        J = self.jacobian(q)
        s = np.linalg.svd(J,compute_uv=False)
        if s[-1] < 1e-10 or s[-1] < 1e-8*s[0]:
            raise ValueError('singular/ill-conditioned Jacobian: exact inverse unavailable')
        return J

    def fk(self, rho, seed=None, tol_m=1e-9, max_step_rad=None):
        """Local bounded FK. seed=previous q for continuous branch tracking."""
        rho = self._rho(rho)
        seed = self.home.copy() if seed is None else self._q(seed)
        if not np.isfinite(tol_m) or tol_m <= 0:
            raise ValueError('tol_m must be positive')
        if max_step_rad is not None and (not np.isfinite(max_step_rad) or max_step_rad <= 0):
            raise ValueError('max_step_rad must be positive')
        # we use continum to get the FK since the sol is non-unique
        fit = least_squares(lambda q:self.geometry(q)[0]-rho, seed,
                            jac=self.jacobian,bounds=(self.lo,self.hi),
                            ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=300)
        residual = np.max(np.abs(fit.fun))
        if not fit.success or residual > tol_m:
            raise ValueError(f'FK failed/unreachable or incompatible data; residual={residual:g} m')
        self._regular_J(fit.x)
        if max_step_rad is not None and np.max(np.abs(fit.x-seed)) > max_step_rad:
            raise ValueError('FK step too large; reduce sample interval/check assembly branch')
        return fit.x

    def inverse_velocity(self, q, qdot):
        return self.jacobian(q)@arr(qdot,(2,),'qdot')

    def forward_velocity(self, q, rhodot):
        return np.linalg.solve(self._regular_J(q),arr(rhodot,(2,),'rhodot'))

    def curvature(self, q, qdot):
        v = arr(qdot,(2,),'qdot')
        return np.einsum('j,ijk,k->i',v,self.hessians(q),v)

    def inverse_acceleration(self, q, qdot, qddot):
        return self.jacobian(q)@arr(qddot,(2,),'qddot')+self.curvature(q,qdot)

    def forward_acceleration(self, q, rhodot, rhoddot):
        v = self.forward_velocity(q,rhodot)
        return np.linalg.solve(self._regular_J(q),arr(rhoddot,(2,),'rhoddot')-self.curvature(q,v))

    def angular_state(self, q, qdot, qddot):
        """Fixed-frame omega, angular acceleration (3-vectors)."""
        _, beta = self._q(q)
        v, a = arr(qdot,(2,),'qdot'),arr(qddot,(2,),'qddot')
        E = np.array([[np.cos(beta),0],[0,1],[-np.sin(beta),0]])
        Edot_v = np.array([-np.sin(beta),0,-np.cos(beta)])*v[0]*v[1]
        return E@v,E@a+Edot_v

    def force_to_torque(self, q, force):
        return self.jacobian(q).T@arr(force,(2,),'force')

    def torque_to_force(self, q, torque):
        return np.linalg.solve(self._regular_J(q).T,arr(torque,(2,),'torque'))

    def actuator_wrench(self, q, force):
        """Actuator resultant force and moment about O; excludes support reactions."""
        f = arr(force,(2,),'force')
        _,_,_,p,u = self.geometry(q)
        forces = f[:,None]*u
        return forces.sum(axis=0),np.cross(p,forces).sum(axis=0)

    @staticmethod
    def _mass(M):
        M = arr(M,(2,2),'M')
        if not np.allclose(M,M.T) or np.min(np.linalg.eigvalsh(M)) <= 0:
            raise ValueError('M must be symmetric positive definite')
        return M

    def inverse_dynamics(self, q, qddot, M, bias, external=(0,0)):
        """bias=c+g+friction, evaluated at current q,qdot by caller."""
        tau = self._mass(M)@arr(qddot,(2,),'qddot')+arr(bias,(2,),'bias')-arr(external,(2,),'external')
        return self.torque_to_force(q,tau)

    def forward_dynamics(self, q, force, M, bias, external=(0,0)):
        rhs = self.force_to_torque(q,force)+arr(external,(2,),'external')-arr(bias,(2,),'bias')
        return np.linalg.solve(self._mass(M),rhs)


def example():
    """default as real wrist, a[0]: left rod, a[1]: right rod; b is vice versa"""
    return Wrist(a=[[-.033,.00,-.2315],[.033,.00,-.2315]],
                 b=[[-.0695/2,-0.02,0],[.0695/2,-0.02,0]],
                 q_min=np.deg2rad([-35,-35]),q_max=np.deg2rad([35,35]),
                 q_home=[0.,0.],rho_min=[.18302,.18302],rho_max=[.28302,.28302],rho_offset=[0.23237,0.23237])


class Verification(unittest.TestCase):
    def test_derivatives_and_roundtrips(self):
        w = example()
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = rng.uniform(-.3,.3,2)
            rho,J,H,_,_ = w.geometry(q)
            eps = 1e-6
            Jfd = np.column_stack([(w.ik(q+eps*e)-w.ik(q-eps*e))/(2*eps) for e in np.eye(2)])
            Hfd = np.stack([(w.jacobian(q+eps*e)-w.jacobian(q-eps*e))/(2*eps) for e in np.eye(2)],axis=2)
            np.testing.assert_allclose(J,Jfd,atol=1e-10)
            np.testing.assert_allclose(H,Hfd,atol=1e-9)
            np.testing.assert_allclose(H,H.transpose(0,2,1),atol=1e-14)
            np.testing.assert_allclose(w.fk(rho),q,atol=1e-7)
            v,a = rng.normal(size=(2,2))
            rv = w.inverse_velocity(q,v)
            ra = w.inverse_acceleration(q,v,a)
            np.testing.assert_allclose(w.forward_velocity(q,rv),v,atol=1e-12)
            np.testing.assert_allclose(w.forward_acceleration(q,rv,ra),a,atol=1e-12)
            dt=1e-4
            rplus=w.ik(q+v*dt+.5*a*dt**2)
            rminus=w.ik(q-v*dt+.5*a*dt**2)
            np.testing.assert_allclose((rplus-2*rho+rminus)/dt**2,ra,atol=3e-8)
            f = rng.normal(size=2)
            tau=w.force_to_torque(q,f)
            self.assertAlmostEqual(float(f@rv),float(tau@v),places=12)
            np.testing.assert_allclose(w.torque_to_force(q,tau),f,atol=1e-12)
            M=np.array([[.1,.01],[.01,.2]])
            force=w.inverse_dynamics(q,a,M,[.2,.3],[.1,0])
            np.testing.assert_allclose(w.forward_dynamics(q,force,M,[.2,.3],[.1,0]),a,atol=1e-12)
            R=rotations(q)[0]
            Rp=rotations(q+eps*v)[0]; Rm=rotations(q-eps*v)[0]
            skew=((Rp-Rm)/(2*eps))@R.T
            omega,acc=w.angular_state(q,v,a)
            np.testing.assert_allclose(omega,[skew[2,1],skew[0,2],skew[1,0]],atol=1e-9)
            op=w.angular_state(q+eps*v,v+eps*a,a)[0]
            om=w.angular_state(q-eps*v,v-eps*a,a)[0]
            np.testing.assert_allclose(acc,(op-om)/(2*eps),atol=1e-9)

    def test_limits_singularity_and_tracking(self):
        w=example()
        with self.assertRaises(ValueError): w.ik([1,0])
        with self.assertRaises(ValueError): w.fk([.21,.21])
        with self.assertRaises(ValueError): w.fk(w.ik([.2,0]),max_step_rad=.01)
        seed=w.home
        for t in np.linspace(0,1,30):
            truth=np.array([.2*t,-.1*t])
            seed=w.fk(w.ik(truth),seed,max_step_rad=.02)
            np.testing.assert_allclose(seed,truth,atol=1e-7)
        singular=Wrist([[.04,0,-.15],[-.04,0,-.15]],[[.04,0,0],[-.04,0,0]],[-.5,-.5],[.5,.5],[0,0])
        with self.assertRaises(ValueError): singular.forward_velocity([0,0],[1,1])


def demo(w):
    q=w.home
    rho=w.ik(q)
    v=np.array([.1,-.08]); a=np.array([.02,.03])
    rv=w.inverse_velocity(q,v); ra=w.inverse_acceleration(q,v,a)
    output={'q_deg':np.rad2deg(q),'rho_m':rho,'stroke_m':w.to_stroke(rho),
            'FK_deg':np.rad2deg(w.fk(rho)), 'J_m_per_rad':w.jacobian(q),
            'H_m_per_rad2':w.hessians(q),'rhodot_m_s':rv,'rhoddot_m_s2':ra,
            'qdot_recovered':w.forward_velocity(q,rv),
            'qddot_recovered':w.forward_acceleration(q,rv,ra),
            'omega_and_angular_acceleration':w.angular_state(q,v,a),
            'force_N_for_tau_1_2_Nm':w.torque_to_force(q,[1,2]),
            'diagnostics':w.diagnostics(q)}
    print(json.dumps(output,default=lambda x:np.asarray(x).tolist(),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--config',help='geometry JSON; SI units')
    args=parser.parse_args()
    if args.test:
        unittest.main(argv=['wrist_tool'],verbosity=2)
    else:
        if args.config:
            with open(args.config,encoding='utf-8') as f: w=Wrist(**json.load(f))
        else: w=example()
        demo(w)
