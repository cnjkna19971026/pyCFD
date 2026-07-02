"""
============================================================
 二維穩態熱傳導 FVM 求解器  (1D -> 2D 退階起點)
============================================================
 控制方程式 :  d/dx(k dT/dx) + d/dy(k dT/dy) + S = 0
 離散式     :  aP*TP = aE*TE + aW*TW + aN*TN + aS*TS + b
 四道面導通 :  aE=aW = k*dy/dx ,  aN=aS = k*dx/dy   (面導通 = k*面積/間距)
 邊界 (Dirichlet): 半格距離 -> 導通加倍,已知值移到 b

 關鍵實作:把二維格子 (i,j) 攤平成一維全域索引 p = j*Nx + i,
           矩陣因此成為「五對角」(對角 + ±1 東西 + ±Nx 南北)。
============================================================
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_2d(Nx, Ny, Lx, Ly, k, Sfun, bc):
    """bc = {'left','right','bottom','top'} 為四面 Dirichlet 溫度。"""
    dx, dy = Lx / Nx, Ly / Ny
    xc = (np.arange(Nx) + 0.5) * dx          # 格子中心 x
    yc = (np.arange(Ny) + 0.5) * dy          # 格子中心 y
    N  = Nx * Ny
    A  = sp.lil_matrix((N, N))               # 稀疏組裝
    b  = np.zeros(N)

    idx = lambda i, j: j * Nx + i            # (i,j) -> 全域索引

    for j in range(Ny):
        for i in range(Nx):
            p = idx(i, j)

            # ---- 東面 ----
            if i < Nx - 1:
                aE = k * dy / dx;  A[p, idx(i+1, j)] = -aE
            else:
                aE = k * dy / (dx/2);  b[p] += aE * bc['right']
            # ---- 西面 ----
            if i > 0:
                aW = k * dy / dx;  A[p, idx(i-1, j)] = -aW
            else:
                aW = k * dy / (dx/2);  b[p] += aW * bc['left']
            # ---- 北面 ----
            if j < Ny - 1:
                aN = k * dx / dy;  A[p, idx(i, j+1)] = -aN
            else:
                aN = k * dx / (dy/2);  b[p] += aN * bc['top']
            # ---- 南面 ----
            if j > 0:
                aS = k * dx / dy;  A[p, idx(i, j-1)] = -aS
            else:
                aS = k * dx / (dy/2);  b[p] += aS * bc['bottom']

            A[p, p] = aE + aW + aN + aS       # aP = 四面之和
            b[p]   += Sfun(xc[i], yc[j]) * (dx * dy)   # 源項 = S * 體積

    T = spla.spsolve(A.tocsr(), b)
    return xc, yc, T.reshape(Ny, Nx), A


# ============================================================
#  Part A — 製造解 (MMS) 收斂驗證
#  exact T = sin(pi x) sin(pi y),  S = 2 k pi^2 sin sin,  四面 T=0
# ============================================================
def part_A():
    Lx = Ly = 1.0
    k = 1.0
    exact = lambda X, Y: np.sin(np.pi*X) * np.sin(np.pi*Y)
    Sfun  = lambda x, y: 2 * k * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    bc = dict(left=0, right=0, bottom=0, top=0)

    print("="*56)
    print(" Part A:  MMS 收斂  T=sin(pi x)sin(pi y)")
    print("="*56)
    print(f"{'Nx=Ny':>7} | {'RMS error':>12} | {'order':>6}")
    print("-"*32)
    prev = None
    for N in [8, 16, 32, 64, 128]:
        xc, yc, T, _ = solve_2d(N, N, Lx, Ly, k, Sfun, bc)
        X, Y = np.meshgrid(xc, yc)
        err = np.sqrt(np.mean((T - exact(X, Y))**2))
        order = "  --" if prev is None else f"{np.log2(prev/err):.2f}"
        print(f"{N:>7} | {err:>12.3e} | {order:>6}")
        prev = err
    print("\n>> 二維 FVM 對純傳導為二階收斂 (order -> 2)")


# ============================================================
#  Part B — 晶片散熱示範 (二維截面)
#  中央方形熱源 (die) + 四周冷卻邊界 (heat sink)
# ============================================================
def part_B():
    Lx = Ly = 0.01           # 10 mm x 10 mm 截面
    k = 150.0                # 矽 ~150 W/m.K
    Nx = Ny = 80
    Tcool = 25.0             # 四周冷卻邊界 (degC)
    Sdie = 1.5e9             # die 區域體積發熱 (W/m^3, 約等於 ~100W die,示意)

    def Sfun(x, y):
        in_die = (0.003 <= x <= 0.007) and (0.003 <= y <= 0.007)
        return Sdie if in_die else 0.0

    bc = dict(left=Tcool, right=Tcool, bottom=Tcool, top=Tcool)
    xc, yc, T, A = solve_2d(Nx, Ny, Lx, Ly, k, Sfun, bc)

    print("\n" + "="*56)
    print(" Part B:  晶片散熱示範 (中央 die + 四周冷卻)")
    print("="*56)
    print(f"  網格         : {Nx} x {Ny}  ({Nx*Ny} 個未知數)")
    print(f"  矩陣非零元素 : {A.tocsr().nnz}  (稠密會是 {(Nx*Ny)**2})")
    print(f"  稀疏度       : {A.tocsr().nnz/(Nx*Ny)**2*100:.3f} %")
    print(f"  最高溫       : {T.max():.2f} degC  (中心 die)")
    print(f"  邊界溫       : {Tcool:.2f} degC")
    return xc, yc, T


if __name__ == "__main__":
    part_A()
    xc, yc, T = part_B()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

        # (1) 溫度場 + 等溫線
        X, Y = np.meshgrid(xc*1e3, yc*1e3)
        pc = ax[0].pcolormesh(X, Y, T, cmap="inferno", shading="auto")
        cs = ax[0].contour(X, Y, T, levels=10, colors="white",
                           linewidths=0.6, alpha=0.7)
        ax[0].clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        ax[0].add_patch(plt.Rectangle((3, 3), 4, 4, fill=False,
                       edgecolor="cyan", lw=1.2, ls="--"))
        ax[0].set_title("Temperature field (degC)")
        ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")
        ax[0].set_aspect("equal")
        fig.colorbar(pc, ax=ax[0], shrink=0.85)

        # (2) 矩陣稀疏結構 (五對角)
        _, _, _, A20 = solve_2d(20, 20, 0.01, 0.01, 150.0,
                                lambda x, y: 0.0,
                                dict(left=25, right=25, bottom=25, top=25))
        ax[1].spy(A20.tocsr(), markersize=1.2)
        ax[1].set_title("Matrix sparsity (20x20 grid)\npenta-diagonal")
        ax[1].set_xlabel("column"); ax[1].set_ylabel("row")

        fig.tight_layout()
        fig.savefig("/mnt/user-data/outputs/heat2d.png", dpi=130)
        print("\n圖已存檔: heat2d.png")
    except Exception as e:
        print("plot skipped:", e)
