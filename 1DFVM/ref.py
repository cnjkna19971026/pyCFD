import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- Problem definition ---
L, k, S      = 0.01, 150.0, -10e8    # m, W/m/K, W/m^3
T_left, T_R  = 300.0, 300.0           # K
N = 20

# N_list = [10,20,40,80,160,320,640,1280]
# numpy support vector calculate
N_list = np.array([10,20,40,80,160,320,640,1280])

def solve_fvm(N):
    dx = L / N
    xc = np.linspace(dx/2, L - dx/2, N)

    aE = aW = k / dx
    aP_int = aE + aW
    aB = k / (dx / 2)          # boundary face coefficient
    bsrc = S * dx

    A = np.zeros((N, N)); b = np.zeros(N)
    for i in range(N):
        if i == 0:                                  # left boundary cell
            A[i, i]   = aB + aE
            A[i, i+1] = -aE
            b[i] = aB * T_left + bsrc
        elif i == N - 1:                            # right boundary cell
            A[i, i-1] = -aW
            A[i, i]   = aW + aB
            b[i] = aB * T_R + bsrc
        else:                                       # interior cell
            A[i, i-1] = -aW
            A[i, i]   = aP_int
            A[i, i+1] = -aE
            b[i] = bsrc

    T = np.linalg.solve(A, b)
    return xc, T

def solve_fvm_error(N):
    dx = L / N
    xc = np.linspace(dx/2, L - dx/2, N)

    aE = aW = k / dx
    aP_int = aE + aW
    aB = k / (dx )          # boundary face coefficient
    bsrc = S * dx

    A = np.zeros((N, N)); b = np.zeros(N)
    for i in range(N):
        if i == 0:                                  # left boundary cell
            A[i, i]   = aB + aE
            A[i, i+1] = -aE
            b[i] = aB * T_left + bsrc
        elif i == N - 1:                            # right boundary cell
            A[i, i]   = aW + aB
            A[i, i-1] = -aW
            b[i] = aB * T_R + bsrc
        else:                                       # interior cell
            A[i, i-1] = -aW
            A[i, i]   = aP_int
            A[i, i+1] = -aE
            b[i] = bsrc

    T = np.linalg.solve(A, b)
    return xc, T

def analytical(x):
    return T_left + (T_R - T_left)*x/L + S/(2*k) * x * (L - x)

# --- Grid convergence study ---
def gridConvStd(N_list,solve_fvm):
    print(f"{'N':>5} {'dx [m]':>12} {'L2 error [K]':>15} {'order':>8}")
    prev_err = None
    for N in N_list:
        xc, T = solve_fvm(N)
        err = np.sqrt(np.mean((T - analytical(xc))**2))
        order = "" if prev_err is None else f"{np.log2(prev_err/err):.2f}"
        print(f"{N:>5} {L/N:>12.2e} {err:>15.4e} {order:>8}")
        prev_err = err

gridConvStd(N_list,solve_fvm)
print("\n")
gridConvStd(N_list,solve_fvm_error)

# --- Plot for N=20 ---
xc, T = solve_fvm(20)
xc_error, T_error = solve_fvm_error(20)
xf = np.linspace(0, L, 400)

#fig1,ax1 = plt.subplots()

#plt.figure()
#plt.plot(xf*1e3, analytical(xf), 'k-', label='Analytical')
#plt.plot(xc*1e3, T, 'ro', ms=6, label='FVM (N=20)')
#plt.plot(xc_error*1e3, T_error, 'ro',linestyle=':', ms=6, label='FVM_error (N=20)')
#plt.xlabel('x [mm]'); plt.ylabel('T [K]'); plt.legend(); plt.grid(True)
#plt.title('1D Steady Conduction with Volumetric Source')


# if you write like this it will show two figure in different wido
#fig1,ax1 = plt.subplots()
#fig2,ax2 = plt.subplots()

fig ,(ax1 , ax2) = plt.subplots(1,2,figsize = (12,5))

ax1.plot(xf*1e3, analytical(xf), 'k-', label='Analytical')
ax2.plot(xf*1e3, analytical(xf), 'k-', label='Analytical')
ax1.plot(xc*1e3, T, 'ro', ms=6, label='FVM (N=20)')
ax2.plot(xc_error*1e3, T_error, 'ro',linestyle=':', ms=6, label='FVM_error (N=20)')
ax1.set_xlabel('x [mm]'); ax1.set_ylabel('T [K]'); ax1.legend(); ax1.grid(True)
ax2.set_xlabel('x [mm]'); ax2.set_ylabel('T [K]'); ax2.legend(); ax2.grid(True)
ax1.set_title('1D Steady Conduction with Volumetric Source')
ax2.set_title('error case')

plt.show()
