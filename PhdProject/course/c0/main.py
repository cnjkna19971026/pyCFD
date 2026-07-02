"""
Lesson 0.1 — 微分如何變成差分,以及「準確度階數」是什麼
策略:拿一個「已知精確導數」的溫度分布當尺,量出三種差分到底準多少。
"""
import numpy as np
import matplotlib.pyplot as plt

# 一個假想的溫度分布(我們"知道"它的精確導數,才能拿來驗證數值微分)
L = 1.0
# this T_func is a test function which we know the result
T_func  = lambda x: 300.0 + 50.0*np.sin(np.pi*x/L)        # 溫度 T(x)
dT_exact= lambda x: 50.0*np.pi/L*np.cos(np.pi*x/L)        # 精確 dT/dx

# 三種有限差分
def forward (f, x, h): return (f(x+h) - f(x))      / h
def backward(f, x, h): return (f(x)   - f(x-h))    / h
def central (f, x, h): return (f(x+h) - f(x-h))    / (2*h)

x0    = 0.30                 # 在這一點測試
exact = dT_exact(x0)
print(f"精確 dT/dx @ x={x0} = {exact:.6f}\n")
print(f"{'Δx':>10} | {'前差':>12} | {'後差':>12} | {'中央差':>12}")
print("-"*56)
for h in [0.1, 0.05, 0.025, 0.0125]:
    print(f"{h:>10.4f} | {forward(T_func,x0,h):>12.5f} | "
          f"{backward(T_func,x0,h):>12.5f} | {central(T_func,x0,h):>12.5f}")
print(f"{'(精確)':>10} | {exact:>12.5f} | {exact:>12.5f} | {exact:>12.5f}")

# 誤差 vs Δx (log-log):斜率 = 準確度階數
hs      = np.logspace(-4, -1, 40)
err_fwd = np.array([abs(forward(T_func, x0, h) - exact) for h in hs])
err_cen = np.array([abs(central(T_func, x0, h) - exact) for h in hs])

# 用最小平方擬合 log-log 斜率,印出"實測階數"
slope_fwd = np.polyfit(np.log(hs), np.log(err_fwd), 1)[0]
slope_cen = np.polyfit(np.log(hs), np.log(err_cen), 1)[0]
print(f"\n實測準確度階數:前差 ≈ {slope_fwd:.2f}(理論 1) | "
      f"中央差 ≈ {slope_cen:.2f}(理論 2)")

# 畫圖
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# (1) 幾何直覺:三條割線去逼近 x0 的切線
xx = np.linspace(0, L, 400)
ax[0].plot(xx, T_func(xx), 'k-', lw=2, label='T(x)')
h_demo = 0.18
ax[0].plot([x0, x0+h_demo], [T_func(x0), T_func(x0+h_demo)], 'b--o', label='forward slope')
ax[0].plot([x0-h_demo, x0], [T_func(x0-h_demo), T_func(x0)], 'g--o', label='backward slope')
ax[0].plot([x0-h_demo, x0+h_demo], [T_func(x0-h_demo), T_func(x0+h_demo)], 'r--o', label='central slope')
ax[0].plot(x0, T_func(x0), 'k*', ms=15)
ax[0].set_title('(1) Finite difference = slope of a secant line')
ax[0].set_xlabel('x'); ax[0].set_ylabel('T (K)'); ax[0].legend(); ax[0].grid(alpha=.3)

# (2) 誤差收斂的 log-log:斜率就是階數
ax[1].loglog(hs, err_fwd, 'b.-', label=f'forward  (slope≈{slope_fwd:.2f})')
ax[1].loglog(hs, err_cen, 'r.-', label=f'central  (slope≈{slope_cen:.2f})')
ax[1].loglog(hs, hs*err_fwd[-1]/hs[-1], 'b:', alpha=.5, label='ref O(Δx)')
ax[1].loglog(hs, hs**2*err_cen[-1]/hs[-1]**2, 'r:', alpha=.5, label='ref O(Δx²)')
ax[1].set_title('(2) Error vs Δx — the slope IS the order of accuracy')
ax[1].set_xlabel('Δx'); ax[1].set_ylabel('|error|'); ax[1].legend(); ax[1].grid(alpha=.3, which='both')

plt.tight_layout()
#plt.savefig('/hom', dpi=110, bbox_inches='tight')
#print("\n圖已存檔 lesson01_result.png")
plt.show()
