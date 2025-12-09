import casadi as csd
import numpy as np

import visualisation as vis
import excavatorModel as mod
from NLPSolver import NLP, Mode, Ts, integrator
from excavatorModel import DutyCycle

print("MPCSimulation_modify: starting simulation...")

# ----------------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------------
TMotor = 0.001              # legacy: motor command interval (现在不再用来积分)
NMotor = int(Ts / TMotor)   # 仅用于打印
TSim = 5.0                  # Simulation time [s]
NSim = int(TSim / Ts)       # Number of MPC steps

print(f"MPCSimulation_modify: Ts = {Ts}, TMotor = {TMotor}")
print(f"MPCSimulation_modify: NMotor = {NMotor}, TSim = {TSim}, NSim = {NSim}")

# ----------------------------------------------------------------------
# Initial state and desired pose (适配新 model0)
# State x = [q1, q2, q3, q1dot, q2dot, q3dot]
# q = [revolute_lift, revolute_tilt, revolute_scoop]
# ----------------------------------------------------------------------

# 初始姿态：全部 0，在新 URDF 的关节范围内
x0 = csd.vertcat(0.0, 0.0, 0.0,   # q
                 0.0, 0.0, 0.0)  # qDot

# 选一个合理的目标关节“档位”
qDesired = csd.DM([0.6, 0.3, -0.5])
poseDesired = mod.forwardKinematics(qDesired)

# ----------------------------------------------------------------------
# Mode / external force / duty cycle
# ----------------------------------------------------------------------
mode = Mode.LIFT
extF_mag = 200.0  # 外力大小 [N]，LIFT 模式向下作用

dutyCycle = DutyCycle.S2_30

if mode != Mode.LIFT:
    extF_mag = 0.0

print(f"MPCSimulation_modify: mode = {mode}, extF = {extF_mag}, dutyCycle = {dutyCycle}")

# ----------------------------------------------------------------------
# 简化版“电机命令”：直接在关节层用 double-integrator 推进一个 Ts
# ----------------------------------------------------------------------
def motorCommands(x, u):
    """
    使用与 NLP 中相同的 integrator 在关节空间推进一个 Ts。

    Parameters
    ----------
    x : casadi.DM(6,1)
        当前状态 [q; qdot]
    u : casadi.DM(3,1)
        当前步关节加速度 qDDot

    Returns
    -------
    x_next : casadi.DM(6,1)
        Ts 秒后的状态。
    """
    x_dm = csd.DM(x)
    u_dm = csd.DM(u)
    return integrator(x_dm, u_dm, Ts)


# ----------------------------------------------------------------------
# Run the simulation loop
# ----------------------------------------------------------------------
xNext = x0

# 构建 NLP（只建一次，循环里复用）
nlp = NLP(mode, extF_mag, dutyCycle)

# 初始可视化帧 (k = 0)
if mode == Mode.LIFT:
    # 初始外力向量 [Fx, Fy] = [0, -extF_mag]
    vis.visualise(xNext, None, qDesired, 0.0, 0, [0, -extF_mag])
else:
    vis.visualise(xNext, None, qDesired, 0.0, 0, None)

# Histories (stored as NumPy arrays for plotting and GUI)
jointAng = np.array(x0[0:3], dtype=float).reshape(3, 1)       # shape (3, 1)
jointVel = np.array(x0[3:6], dtype=float).reshape(3, 1)       # shape (3, 1)
jointAcc = np.array([[0.0, 0.0, 0.0]]).T                      # shape (3, 1)
motorTorque = np.array([[0.0, 0.0, 0.0]]).T                   # shape (3, 1)
motorVel = np.array([[0.0, 0.0, 0.0]]).T                      # shape (3, 1)
motorPower = np.array([[0.0, 0.0, 0.0]]).T                    # shape (3, 1)

print("MPCSimulation_modify: entering main loop...")

for k in range(NSim):  # Run simulation for NSim * Ts = TSim seconds

    # Solve MPC from current state xNext toward poseDesired
    sol = nlp.solveNLP(xNext, poseDesired)

    # Optimal control at current step (joint accelerations)
    u0 = sol.value(nlp.u[:, 0])  # shape (3,)
    u0_dm = csd.DM(u0)

    # Propagate state over one MPC step using joint-space integrator
    xNext = motorCommands(xNext, u0_dm)

    # Visualise current configuration
    if mode == Mode.LIFT:
        # 外力变量在 NLP 里叫 F_ext，形状 (2, N)
        extF_vec = sol.value(nlp.F_ext[:, 0])  # 2D force (Fx, Fy)
        vis.visualise(xNext, None, qDesired, (k + 1) * Ts, k + 1, extF_vec)
    else:
        vis.visualise(xNext, None, qDesired, (k + 1) * Ts, k + 1, None)

    # ---- Log data for plotting / GUI ----
    qk = np.array(xNext[0:3], dtype=float).reshape(3, 1)
    qdotk = np.array(xNext[3:6], dtype=float).reshape(3, 1)
    uk = np.array(u0, dtype=float).reshape(3, 1)

    jointAng = np.hstack((jointAng, qk))
    jointVel = np.hstack((jointVel, qdotk))
    jointAcc = np.hstack((jointAcc, uk))

    # ---- Motor torque / velocity / power (用于事后分析，不进入 MPC 约束) ----
    if mode == Mode.LIFT:
        F_ext_vec = csd.vertcat(0, -extF_mag)
    else:
        F_ext_vec = csd.vertcat(0, 0)

    tau_dm = mod.motorTorque(xNext[0:3], xNext[3:6], uk[:, 0], F_ext_vec)
    tau = np.array(tau_dm, dtype=float).reshape(3, 1)

    w_dm = mod.motorVel(xNext[0:3], xNext[3:6])
    w = np.array(w_dm, dtype=float).reshape(3, 1)

    motorTorque = np.hstack((motorTorque, tau))
    motorVel = np.hstack((motorVel, w))

    # Instantaneous motor power in kW
    p = np.abs(tau[:, 0] * w[:, 0])[:, np.newaxis] / 1000.0
    motorPower = np.hstack((motorPower, p))

    if k % 5 == 0 or k == NSim - 1:
        print(f"MPCSimulation_modify: step {k+1}/{NSim} done")

print("MPCSimulation_modify: main loop finished, plotting & saving...")

# ----------------------------------------------------------------------
# Save MPC data for GUI playback
# ----------------------------------------------------------------------
time_vec = np.linspace(0.0, TSim, jointAng.shape[1])

np.savez(
    "mpc_results.npz",
    jointAng=jointAng,
    jointVel=jointVel,
    jointAcc=jointAcc,
    motorTorque=motorTorque,
    motorVel=motorVel,
    motorPower=motorPower,
    time=time_vec,
    Ts=Ts,
    TSim=TSim,
)

print("MPCSimulation_modify: saved MPC data to mpc_results.npz")

# ----------------------------------------------------------------------
# Plots and video using visualisation.py
# ----------------------------------------------------------------------
vis.plotMotorOpPt(motorTorque, motorVel, dutyCycle)
vis.graph(0.0, TSim, Ts, "Joint Angles", r"$\mathsf{q\ (rad)}$", x=jointAng)
vis.graph(0.0, TSim, Ts, "Joint Angular Velocities", r"$\mathsf{\dot{q}\ (rad\ s^{-1})}$", x=jointVel)
vis.graph(0.0, TSim, Ts, "Joint Angular Accelerations", r"$\mathsf{\ddot{q}\ (rad\ s^{-2})}$", u=jointAcc)
vis.graph(0.0, TSim, Ts, "Motor Torques", "Motor torque (Nm)", motorTorque=motorTorque)
vis.graph(0.0, TSim, Ts, "Motor Velocities", r"Motor velocity ($\mathsf{rad\ s^{-1}}$)", motorVel=motorVel)
vis.graph(0.0, TSim, Ts, "Motor Powers", "Motor power (kW)", motorPower=motorPower)

print("MPCSimulation_modify: creating video...")
vis.createVideo(0, NSim, "Excavator", int(1.0 / Ts))

print("MPCSimulation_modify: Done.")
print(f"Results saved in folder: {vis.visFolder} and file mpc_results.npz")
