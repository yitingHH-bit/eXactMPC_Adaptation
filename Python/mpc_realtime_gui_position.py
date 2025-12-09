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


# ============ 几何常量 ============
LEN_BA = float(C.lenBA)   # base → boom
LEN_AL = float(C.lenAL)   # boom → arm
LEN_LM = float(C.lenLM)   # arm → tip
TOTAL_LEN = LEN_BA + LEN_AL + LEN_LM


# ============ 简化版 motor_commands：在关节层用 integrator 推进 ============
def motor_commands(x, u):
    """
    使用与 NLP 中相同的 integrator 在关节空间推进一个 Ts。

    x : casadi.DM(6,1) 或 6 元 array  [q; qDot]
    u : casadi.DM(3,1) 或 3 元 array  关节加速度
    """
    x_dm = csd.DM(x)
    u_dm = csd.DM(u)
    return integrator(x_dm, u_dm, Ts)


class ExcavatorRealtimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Excavator MPC – Realtime 3D Simulation")

        # ========== 3D 画布 ==========
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ========== 控件区域 ==========
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # 左侧：控制按钮
        btn_frame = ttk.Frame(controls)
        btn_frame.pack(side=tk.LEFT)

        self.start_button = ttk.Button(
            btn_frame, text="▶ Start (Realtime MPC)", command=self.start_simulation
        )
        self.start_button.grid(row=0, column=0, padx=3)

        self.stop_button = ttk.Button(
            btn_frame, text="⏸ Stop", command=self.stop_simulation
        )
        self.stop_button.grid(row=0, column=1, padx=3)

        self.reset_button = ttk.Button(
            btn_frame, text="⟲ Reset", command=self.reset_simulation
        )
        self.reset_button.grid(row=0, column=2, padx=3)

        # 中间：状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_label = ttk.Label(controls, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # 右侧：末端目标坐标输入（X, Z, Theta）
        target_frame = ttk.LabelFrame(controls, text="Target pose (end-effector)")
        target_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(target_frame, text="X [m]:").grid(row=0, column=0, sticky="e")
        ttk.Label(target_frame, text="Z [m]:").grid(row=1, column=0, sticky="e")
        ttk.Label(target_frame, text="Theta [rad]:").grid(row=2, column=0, sticky="e")

        # ---- MPC / 仿真状态 ----
        self.mode = Mode.LIFT
        self.extF = 200.0  # LIFT 模式的向下外力
        self.dutyCycle = DutyCycle.S2_30

        self.TSim = 5.0
        self.NSim = int(self.TSim / Ts)

        # 初始关节姿态 x = [q, qdot]
        self.x0 = csd.vertcat(0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0)

        # 默认目标关节“档位”和对应末端姿态
        self.qDesired = csd.DM([0.6, 0.3, -0.5])
        self.poseDesired = mod.forwardKinematics(self.qDesired)

        # 目标输入框默认值
        self.target_x_var = tk.StringVar(value=f"{float(self.poseDesired[0]):.3f}")
        self.target_z_var = tk.StringVar(value=f"{float(self.poseDesired[1]):.3f}")
        self.target_theta_var = tk.StringVar(value=f"{float(self.poseDesired[2]):.3f}")

        ttk.Entry(target_frame, width=8,
                  textvariable=self.target_x_var).grid(row=0, column=1, padx=2)
        ttk.Entry(target_frame, width=8,
                  textvariable=self.target_z_var).grid(row=1, column=1, padx=2)
        ttk.Entry(target_frame, width=8,
                  textvariable=self.target_theta_var).grid(row=2, column=1, padx=2)

        self.update_target_button = ttk.Button(
            target_frame, text="Update Target",
            command=self.update_target_from_inputs
        )
        self.update_target_button.grid(row=0, column=2, rowspan=3, padx=5, pady=2)

        # NLP 对象（只建一次，在循环里重复 solve）
        self.nlp = NLP(self.mode, self.extF, self.dutyCycle)

        # 仿真状态
        self.running = False
        self.k = 0
        self.x = self.x0

        # 初始画一帧
        q0 = np.array(self.x[0:3], dtype=float).reshape(-1)
        self.draw_frame(q0, t=0.0)

    # ---------- 运动学：计算关节点 3D 坐标 ----------
    def link_positions_3d(self, q):
        """
        根据关节角 q = [alpha, beta, gamma] 计算 4 个关节点的 3D 坐标。
        由于是平面机构，我们令 Y=0，在 X–Z 平面内运动。
        """
        alpha, beta, gamma = float(q[0]), float(q[1]), float(q[2])

        p0 = np.array([0.0, 0.0, 0.0])

        p1 = p0 + np.array(
            [LEN_BA * math.cos(alpha), 0.0, LEN_BA * math.sin(alpha)]
        )

        p2 = p1 + np.array(
            [LEN_AL * math.cos(alpha + beta), 0.0,
             LEN_AL * math.sin(alpha + beta)]
        )

        p3 = p2 + np.array(
            [LEN_LM * math.cos(alpha + beta + gamma), 0.0,
             LEN_LM * math.sin(alpha + beta + gamma)]
        )

        return np.vstack([p0, p1, p2, p3])

    # ---------- 画一帧 ----------
    def draw_frame(self, q, t):
        pts = self.link_positions_3d(q)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

        self.ax.cla()

        # 当前臂
        self.ax.plot(X, Y, Z, "-o", label="excavator arm")

        # 期望末端骨架（如果 IK 成功）
        try:
            qd_np = np.array(self.qDesired, dtype=float).reshape(-1)
            pts_d = self.link_positions_3d(qd_np)
            Xd, Yd, Zd = pts_d[:, 0], pts_d[:, 1], pts_d[:, 2]
            self.ax.plot(Xd, Yd, Zd, "--o", label="desired", alpha=0.7)
        except Exception:
            pass

        # 地面
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

    # ---------- 根据输入框更新目标末端姿态 ----------
    def update_target_from_inputs(self):
        """
        从 GUI 输入框读取 (X, Z, Theta)，更新 poseDesired 和 qDesired。
        这里就先做一次 IK + NaN 检查，尽量把“不可达目标”挡在 MPC 之前。
        """
        import numpy as np

        try:
            x = float(self.target_x_var.get())
            z = float(self.target_z_var.get())
            theta_str = self.target_theta_var.get().strip()

            if theta_str == "":
                theta = float(self.poseDesired[2])
            else:
                theta = float(theta_str)

            new_pose = csd.DM([x, z, theta])

            # 先试着做一次 IK
            q_d = mod.inverseKinematics(new_pose)
            q_np = np.array(q_d, dtype=float).reshape(-1)

            if not np.all(np.isfinite(q_np)):
                raise ValueError(
                    "IK returned NaN/Inf — target is likely outside workspace "
                    "or violates joint limits."
                )

            # 通过检查：更新内部目标
            self.poseDesired = new_pose
            self.qDesired = q_d

            self.status_var.set(
                f"Updated target: X={x:.2f} m, Z={z:.2f} m, θ={theta:.2f} rad"
            )

        except Exception as e:
            # 保持原来的目标不变，只提示错误
            self.status_var.set(f"Invalid / unreachable target: {e}")

    # ---------- 单步 MPC 更新 ----------
    def step_mpc(self):
        """
        做一步 MPC：
        1) 用当前 x 调 nlp.solveNLP
        2) 取 u0
        3) 用 integrator 推进到 x_{k+1}
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
