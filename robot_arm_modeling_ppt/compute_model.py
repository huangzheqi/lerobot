import numpy as np
from scipy import signal

np.set_printoptions(precision=3, suppress=True)

# ---- physical params (2-link planar arm + grasped payload mp) ----
m1, m2, mp = 2.0, 1.0, 0.5
l1, l2 = 0.5, 0.4
lc1, lc2 = 0.25, 0.2
I1, I2 = 0.05, 0.02
fv = np.array([0.5, 0.3])
g = 9.81

# equilibrium: hanging grasp configuration q0=[-pi/2, 0], dq0=0 -> tau0=G(q0)=0
q0 = np.array([-np.pi/2, 0.0])
c2 = np.cos(q0[1]); s1 = np.sin(q0[0]); s12 = np.sin(q0[0]+q0[1])

M11 = m1*lc1**2 + I1 + I2 + m2*(l1**2+lc2**2+2*l1*lc2*c2) + mp*(l1**2+l2**2+2*l1*l2*c2)
M12 = I2 + m2*(lc2**2+l1*lc2*c2) + mp*(l2**2+l1*l2*c2)
M22 = I2 + m2*lc2**2 + mp*l2**2
M = np.array([[M11, M12],[M12, M22]])

h1 = (m1*lc1 + (m2+mp)*l1)*g
h2 = (m2*lc2 + mp*l2)*g
G0_1 = h1*np.cos(q0[0]) + h2*np.cos(q0[0]+q0[1])   # gravity torque at q0 (should be 0)
G0_2 = h2*np.cos(q0[0]+q0[1])
Kg = np.array([[-h1*s1-h2*s12, -h2*s12],[-h2*s12, -h2*s12]])
Fv = np.diag(fv)

print("M(q0) =\n", M)
print("tau0 = G(q0) =", np.array([G0_1, G0_2]))
print("Kg =\n", Kg)
Minv = np.linalg.inv(M)
print("Minv =\n", Minv)

n = 2
A = np.block([[np.zeros((n,n)), np.eye(n)], [-Minv@Kg, -Minv@Fv]])
B = np.vstack([np.zeros((n,n)), Minv])
C = np.hstack([np.eye(n), np.zeros((n,n))])
D = np.zeros((n,n))
print("\nA =\n", A)
print("B =\n", B)

# transfer function from input 1 (tau1)
num, den = signal.ss2tf(A, B, C, D, input=0)
print("\nss2tf input tau1:")
print("num q1 <- tau1:", num[0])
print("num q2 <- tau1:", num[1])
print("den:", den)

poles = np.linalg.eigvals(A)
poles = poles[np.argsort(-poles.real)]
print("\npoles:", np.array2string(poles, precision=3))
print("max real part:", poles.real.max())

# pretty fraction for G11
nm = num[0]
print("\nG11(s) = ({:.3f} s^2 + {:.3f} s + {:.2f}) / (s^4 + {:.2f} s^3 + {:.1f} s^2 + {:.1f} s + {:.1f})".format(nm[2], nm[3], nm[4], den[1], den[2], den[3], den[4]))

# DC gain check: G11(0) = Kg^{-1}[0,0]
print("G11(0) =", nm[4]/den[4], " | inv(Kg)[0,0] =", np.linalg.inv(Kg)[0,0])
