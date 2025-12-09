import casadi as csd
from enum import Enum
import excavatorConstants as C
import excavatorModel as mod

# ---------------------------------------------------------------------------
# Prediction horizon
# ---------------------------------------------------------------------------
T = 2.0          # [s]
N = 20           # number of intervals
Ts = T / N       # sampling time

# Debug 标志：True 时忽略电机相关的约束（扭矩/转速/功率），先保证 MPC 能跑通
DEBUG_IGNORE_MOTOR_LIMITS = True  # 先设 True 调试


# ---------------------------------------------------------------------------
# Modes for external force behaviour
# ---------------------------------------------------------------------------
class Mode(Enum):
    NO_LOAD = 0   # No external forces
    LIFT = 1      # External downward force in the bucket
    DIG = 2       # Resistive force against bucket velocity (not implemented)


# ---------------------------------------------------------------------------
# Simple joint-space double-integrator used inside MPC
# x = [q; qDot], u = qDDot
# ---------------------------------------------------------------------------
def integrator(x, u, t):
    return csd.vertcat(
        x[0] + x[3] * t + 0.5 * u[0] * t**2,
        x[1] + x[4] * t + 0.5 * u[1] * t**2,
        x[2] + x[5] * t + 0.5 * u[2] * t**2,
        x[3] + u[0] * t,
        x[4] + u[1] * t,
        x[5] + u[2] * t,
    )

# ---------------------------------------------------------------------------
# Nonlinear MPC problem (CasADi Opti)
# ---------------------------------------------------------------------------
class NLP:
    def __init__(self, mode: Mode, extF_mag: float, dutyCycle):
        """
        mode      : Mode.NO_LOAD / Mode.LIFT / Mode.DIG
        extF_mag  : magnitude of external force [N] (used in LIFT mode, downward)
        dutyCycle : excavatorModel.DutyCycle enum for motor torque limits
        """
        self.mode = mode
        self.extF_mag = extF_mag
        self.dutyCycle = dutyCycle

        self.opti = csd.Opti()

        # Decision variables
        self.x = self.opti.variable(6, N + 1)   # state: [q; qDot]
        self.u = self.opti.variable(3, N)       # input: qDDot

        # Parameters
        self.x0 = self.opti.parameter(6, 1)      # initial state
        self.poseDesired = self.opti.parameter(3, 1)  # desired tip pose [x, y, theta]

        # Extra variables (for convenience, to avoid repeated computations)
        self.poseActual = self.opti.variable(3, N)     # forward kinematics
        self.F_ext = self.opti.variable(2, N)          # external force at bucket tip [Fx, Fy]
        self.motorTorque = self.opti.variable(3, N)
        self.motorVel = self.opti.variable(3, N)
        self.motorPower = self.opti.variable(3, N)

        # ------------------------------------------------------------------
        # Weights in cost function
        # ------------------------------------------------------------------
        xWeight = 1.0
        yWeight = 1.0
        # Scale orientation error by tip length
        thetaWeight = C.lenLM**2
        terminalWeight = 1.0
        regularisationWeight = 0.01

        # Joint limits for the new model0 URDF
        # q = [revolute_lift, revolute_tilt, revolute_scoop]
        q_max = csd.vertcat(1.6057029, 1.4835297, 1.9198622)   # [rad]
        q_min = csd.vertcat(0.0,       -0.4886922, -1.2217305) # [rad]

        # ------------------------------------------------------------------
        # Build objective and constraints ONCE
        # ------------------------------------------------------------------
        L = 0

        for k in range(N):
            # Tracking cost (tip pose vs desired pose)
            L += (
                xWeight * (self.poseActual[0, k] - self.poseDesired[0])**2
                + yWeight * (self.poseActual[1, k] - self.poseDesired[1])**2
                + thetaWeight * (self.poseActual[2, k] - self.poseDesired[2])**2
            )

            # Regularisation on joint accelerations
            L += regularisationWeight * (
                self.u[0, k]**2 + self.u[1, k]**2 + self.u[2, k]**2
            )

            # ---- System dynamics ----
            self.opti.subject_to(
                self.x[:, k + 1] == integrator(self.x[:, k], self.u[:, k], Ts)
            )

            # ---- Forward kinematics ----
            self.opti.subject_to(
                self.poseActual[:, k] == mod.forwardKinematics(self.x[0:3, k + 1])
            )

            # ---- External force ----
            if self.mode == Mode.LIFT:
                # Downward load: Fy = -extF_mag
                self.opti.subject_to(
                    self.F_ext[:, k] == csd.vertcat(0, -self.extF_mag)
                )
            # elif self.mode == Mode.DIG:
            #     # TODO: force opposite to tip velocity
            #     ...
            else:
                # NO_LOAD or default
                self.opti.subject_to(
                    self.F_ext[:, k] == csd.vertcat(0, 0)
                )

            # ---- Joint limits ----
            self.opti.subject_to(self.x[0:3, k + 1] <= q_max)
            self.opti.subject_to(self.x[0:3, k + 1] >= q_min)

            # ==================================================================
            # Motor 部分：根据 DEBUG_IGNORE_MOTOR_LIMITS 决定是否添加约束
            # ==================================================================
            if not DEBUG_IGNORE_MOTOR_LIMITS:
                # ---- Motor torque & speed ----
                self.opti.subject_to(
                    self.motorTorque[:, k]
                    == mod.motorTorque(
                        self.x[0:3, k],        # q
                        self.x[3:6, k],        # qDot
                        self.u[:, k],          # qDDot
                        self.F_ext[:, k],      # external tip force (2x1)
                    )
                )
                self.opti.subject_to(
                    self.motorVel[:, k]
                    == mod.motorVel(self.x[0:3, k], self.x[3:6, k])
                )
                self.opti.subject_to(
                    self.motorPower[:, k]
                    == self.motorTorque[:, k] * self.motorVel[:, k]
                )

                # ---- Motor speed limits (±471.24 rad/s ≈ 4500 rpm) ----
                v_lim = 471.24
                self.opti.subject_to(
                    self.motorVel[:, k] <= csd.vertcat(v_lim, v_lim, v_lim)
                )
                self.opti.subject_to(
                    self.motorVel[:, k] >= csd.vertcat(-v_lim, -v_lim, -v_lim)
                )

                # ---- Motor torque limits (duty-cycle dependent) ----
                tau_lim_pos = mod.motorTorqueLimit(self.motorVel[:, k], self.dutyCycle)
                tau_lim_neg = mod.motorTorqueLimit(-self.motorVel[:, k], self.dutyCycle)

                self.opti.subject_to(self.motorTorque[:, k] <= tau_lim_pos)
                self.opti.subject_to(self.motorTorque[:, k] >= -tau_lim_pos)
                self.opti.subject_to(self.motorTorque[:, k] <= tau_lim_neg)
                self.opti.subject_to(self.motorTorque[:, k] >= -tau_lim_neg)

                # ---- Motor power limits (various sign combinations) ----
                motorPowerLim = 8250.0  # [W]
                P0 = self.motorPower[0, k]
                P1 = self.motorPower[1, k]
                P2 = self.motorPower[2, k]

                self.opti.subject_to(P0 + P1 + P2 <= motorPowerLim)
                self.opti.subject_to(P0 + P1 - P2 <= motorPowerLim)
                self.opti.subject_to(P0 - P1 + P2 <= motorPowerLim)
                self.opti.subject_to(P0 - P1 - P2 <= motorPowerLim)
                self.opti.subject_to(-P0 + P1 + P2 <= motorPowerLim)
                self.opti.subject_to(-P0 + P1 - P2 <= motorPowerLim)
                self.opti.subject_to(-P0 - P1 + P2 <= motorPowerLim)
                self.opti.subject_to(-P0 - P1 - P2 <= motorPowerLim)
            else:
                # Debug 模式：不考虑电机动力学/约束，把它们固定为 0，避免自由变量发散
                self.opti.subject_to(
                    self.motorTorque[:, k] == csd.DM.zeros(3, 1)
                )
                self.opti.subject_to(
                    self.motorVel[:, k] == csd.DM.zeros(3, 1)
                )
                self.opti.subject_to(
                    self.motorPower[:, k] == csd.DM.zeros(3, 1)
                )

        # Terminal cost on joint velocities (prefer stopping)
        L += terminalWeight * (
            self.x[3, N]**2 + self.x[4, N]**2 + self.x[5, N]**2
        )

        self.opti.minimize(L)

        # Initial state constraint  —— 只加一次！
        self.opti.subject_to(self.x[:, 0] == self.x0)

        # Solver options  —— 只在 __init__ 里设置一次
        opts = {}
        opts["verbose_init"] = False
        opts["verbose"] = False
        opts["print_time"] = False
        opts["ipopt.print_level"] = 0

        self.opti.solver("ipopt", opts)

    # ------------------------------------------------------------------
    # Solve the NLP for given initial state and desired tip pose
    # ------------------------------------------------------------------
    def solveNLP(self, x0, poseDesired):
        # Set parameter values
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.poseDesired, poseDesired)

        # Initial guess for state trajectory
        qDesired = mod.inverseKinematics(poseDesired)

        for k in range(N):
            alpha = (k + 1) / N
            q0 = x0[0:3]
            qGuess = alpha * (qDesired - q0) + q0
            qDotGuess = (qDesired - q0) / T

            self.opti.set_initial(
                self.x[:, k],
                csd.vertcat(qGuess, qDotGuess),
            )

        # 最后一帧给个简单初值
        self.opti.set_initial(
            self.x[:, N],
            csd.vertcat(qDesired, csd.DM.zeros(3, 1)),
        )

        # Solve
        sol = self.opti.solve()
        return sol
