import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import casadi as csd
import excavatorConstants as C
import excavatorModel as mod
from excavatorModel import DutyCycle
from NLPSolver import NLP, Mode, Ts, integrator


# ============================================================================
# 几何常量（直接从新的 excavatorConstants/model0 读）
# q = [alpha, beta, gamma] = [revolute_lift, revolute_tilt, revolute_scoop]
# ============================================================================
LEN_BA = float(C.lenBA)   # boom link length (mountboom -> liftboom -> tilt joint)
LEN_AL = float(C.lenAL)   # arm link length  (tiltboom -> scoop joint)
LEN_LM = float(C.lenLM)   # bucket tip length (scoop joint -> tip)
TOTAL_LEN = LEN_BA + LEN_AL + LEN_LM


# ======================================================================
# 用关节层二阶积分器直接推进状态（不再走 actuator / jointAngles）
# ======================================================================
def motor_commands(x, u):
    """
    简化版本：用 NLP 里同一个 integrator 在关节层推进一个 Ts。
    x: 当前状态 [q; qDot] (6x1 或 6,)
    u: 当前步关节加速度 qDDot (3x1 或 3,)
    """
    x_dm = csd.DM(x)
    u_dm = csd.DM(u)
    return integrator(x_dm, u_dm, Ts)


class ExcavatorRealtimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Excavator MPC – Realtime 3D Simulation (model0)")

        # ========== 3D 画布 ==========
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ========== 控件区域 ==========
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(
            controls, text="▶ Start (Realtime MPC)", command=self.start_simulation
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            controls, text="⏸ Stop", command=self.stop_simulation
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(
            controls, text="⟲ Reset", command=self.reset_simulation
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_label = ttk.Label(controls, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # ========== MPC / 仿真状态 ==========
        self.mode = Mode.LIFT
        self.extF = 100.0  # LIFT 模式的外力（小臂比较小，200 N 先试试）
        self.dutyCycle = DutyCycle.S2_30

        self.TSim = 15.0
        self.NSim = int(self.TSim / Ts)

        # 选一个在关节限制内的目标关节角，然后用 FK 算出来目标末端位姿
        q_desired0 = csd.DM([0.6, 0.3, -0.5])  # 合理、稍微抬起的姿态
        pose_desired0 = mod.forwardKinematics(q_desired0)

        self.poseDesired = pose_desired0
        self.qDesired = q_desired0

        # 初始状态 x = [q, qdot]（全 0，在新 URDF 的 joint limits 里）
        self.x0 = csd.vertcat(
            0.0, 0.0, 0.0,   # q
            0.0, 0.0, 0.0    # qDot
        )

        self.running = False
        self.k = 0
        self.x = self.x0

        # ====== 关键修改：仅在这里 new 一次 NLP，对应新的 model0 ======
        self.nlp = NLP(self.mode, self.extF, self.dutyCycle)

        # 初始画一帧
        q0 = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q0, t=0.0)

    # ---------- 运动学：计算关节点 3D 坐标 ----------
    def link_positions_3d(self, q):
        # q = [alpha, beta, gamma] = [revolute_lift, revolute_tilt, revolute_scoop]
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])

        p0 = np.array([0.0, 0.0, 0.0])

        p1 = p0 + np.array(
            [LEN_BA * math.cos(alpha),
             0.0,
             LEN_BA * math.sin(alpha)]
        )

        p2 = p1 + np.array(
            [LEN_AL * math.cos(alpha + beta),
             0.0,
             LEN_AL * math.sin(alpha + beta)]
        )

        p3 = p2 + np.array(
            [LEN_LM * math.cos(alpha + beta + gamma),
             0.0,
             LEN_LM * math.sin(alpha + beta + gamma)]
        )

        return np.vstack([p0, p1, p2, p3])

    # ---------- 画一帧 ----------
    def draw_frame(self, q, t):
        pts = self.link_positions_3d(q)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

        self.ax.cla()

        self.ax.plot(X, Y, Z, "-o", label="excavator arm")

        L = TOTAL_LEN + 0.5
        self.ax.plot([0, L], [0, 0], [0, 0], "k--", linewidth=1, label="ground")

        self.ax.set_xlim(0, L)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(-0.5, L)

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title(f"Realtime MPC – t = {t:.2f} s  (k = {self.k})")

        self.ax.view_init(elev=25, azim=-60)
        self.ax.legend(loc="upper left", fontsize=8)

        self.canvas.draw()

    # ---------- 单步 MPC 更新 ----------
    # ---------- 单步 MPC 更新 ----------
    # ---------- 单步 MPC 更新 ----------
    def step_mpc(self):
        """
        做一步 MPC：
        1) 用当前 x 调 nlp.solveNLP
        2) 取 u0
        3) 用 integrator (motor_commands) 推进到 x_{k+1}
        """
        if self.k >= self.NSim:
            self.status_var.set("Simulation finished (k reached NSim).")
            return False  # 仿真结束

        try:
            sol = self.nlp.solveNLP(self.x, self.poseDesired)
        except Exception as e:
            import traceback
            print(f"\n=== NLP solve failed at step k={self.k} ===")
            traceback.print_exc()
            self.status_var.set(f"NLP error at k={self.k}: {e}")
            return False

        # 当前步的加速度 u0（第一步控制）
        u0 = sol.value(self.nlp.u[:, 0])    # numpy(3,)
        u0_dm = csd.DM(u0)

        # 用关节层 double-integrator 推进一个 Ts
        self.x = motor_commands(self.x, u0_dm)

        self.k += 1
        return True

    # ---------- 动画循环（实时仿真） ----------
    def update_frame(self):
        if not self.running:
            return

        ok = self.step_mpc()
        if not ok:
            # 不覆盖 step_mpc 写进去的 status
            self.running = False
            self.start_button.config(text="▶ Start (Realtime MPC)")
            return

        t = self.k * Ts
        q = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q, t)

        # 刷新间隔（毫秒）
        self.root.after(10, self.update_frame)


    # ---------- 控件回调 ----------
    def start_simulation(self):
        if self.running:
            return
        self.status_var.set("Running realtime MPC...")
        self.running = True
        self.start_button.config(text="▶ Running...")
        self.update_frame()

    def stop_simulation(self):
        if not self.running:
            return
        self.running = False
        self.status_var.set("Paused.")
        self.start_button.config(text="▶ Start (Realtime MPC)")

    def reset_simulation(self):
        if self.running:
            self.stop_simulation()
        self.k = 0
        self.x = self.x0
        q0 = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q0, t=0.0)
        self.status_var.set("Reset to initial state.")


def main():
    root = tk.Tk()
    app = ExcavatorRealtimeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
