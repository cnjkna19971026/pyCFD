import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class snapshotrecorder:

    def __init__(self, stride = 1):
        self.snap = []
        self.stride = stride
    def __call__(self,literal ,field):
        if literal % self.stride == 0:
            self.snap.append(field.copy())

class probRec:
    def __init__(self,i_probe, dt):
        self.i_probe = i_probe
        self.dt = dt
        self.t_hist = [ ]
        self.T_hist = [ ]

    def __call__ (self,n ,field):
        self.t_hist.append(n*self.dt) 
        self.T_hist.append(field[self.i_probe])
            
def plt_proRec(t_hist,T_hist):
    plt.plot(t_hist,T_hist)
    plt.xlabel("Time(s)")
    plt.xlabel("Temperature(C)")
    plt.show()



def animation_snapshot(x, snap, 
                      filename="evolution.gif",
                      xlabel="x (mm)", 
                      ylabel="T (°C)",
                      title_fmt="niter = {n}",
                      xscale=1e3,
                      interval=80, 
                      fps=15,
                      ymargin=5):
    """
    通用一維場量演化動畫產生器。

    參數:
        x          : 空間座標陣列
        snap       : 快照 list，每個元素是某時間步的解陣列
        filename   : 輸出 GIF 檔名
        xlabel     : x 軸標籤
        ylabel     : y 軸標籤
        title_fmt  : 標題格式字串，{n} 會被替換成幀編號
        xscale     : x 軸縮放係數（m → mm 用 1e3；不縮放用 1）
        interval   : 每幀間隔（毫秒）
        fps        : GIF 播放速率
        ymargin    : y 軸上下留白
    """
    # 固定 y 軸範圍，否則每幀重縮放會看起來沒在動
    ymin = min(s.min() for s in snap)
    ymax = max(s.max() for s in snap)

    fig, ax = plt.subplots()
    line, = ax.plot(x * xscale, snap[0], 'o-', ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin - ymargin, ymax + ymargin)
    title = ax.set_title(title_fmt.format(n=0))

    def update(n):
        line.set_ydata(snap[n])
        title.set_text(title_fmt.format(n=n))
        return line, title

    anim = FuncAnimation(fig, update, frames=len(snap),
                         interval=interval, blit=False)
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)          # 避免殘留圖窗佔記憶體
    print(f"[animate] 已儲存 {filename}（共 {len(snap)} 幀）")
                       

