import casadi as csd
from enum import Enum
import excavatorConstants as C

"""
Excavator dynamics & kinematics model for MPC.

⚠️ Joint convention for the new URDF model `model0`:

    q = [alpha, beta, gamma]

    alpha -> URDF joint 'revolute_lift'   (mountboom -> liftboom)
    beta  -> URDF joint 'revolute_tilt'   (liftboom  -> tiltboom)
    gamma -> URDF joint 'revolute_scoop'  (tiltboom  -> scoop)

Geometry, link masses and inertias are defined in `excavatorConstants.py`
and have been updated to match the `model0` URDF (liftboom, tiltboom, scoop).
"""


class DutyCycle(Enum):
    S1 = 0
    S2_60 = 1
    S2_30 = 2
    PEAK = 3


def q_from_urdf(joint_vec):
    """
    Map URDF joint order [revolute_lift, revolute_tilt, revolute_scoop]
    to internal generalized coordinates q = [alpha, beta, gamma].

    joint_vec: iterable/list/DM with length 3
    """
    return csd.vertcat(joint_vec[0], joint_vec[1], joint_vec[2])


def forwardKinematics(q):
    """
    Planar forward kinematics (2D tip pose) for the 3-DOF arm.
    Uses link lengths from excavatorConstants (lenBA, lenAL, lenLM).
    """
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    x = (
        C.lenBA * csd.cos(alpha)
        + C.lenAL * csd.cos(alpha + beta)
        + C.lenLM * csd.cos(alpha + beta + gamma)
    )
    y = (
        C.lenBA * csd.sin(alpha)
        + C.lenAL * csd.sin(alpha + beta)
        + C.lenLM * csd.sin(alpha + beta + gamma)
    )
    theta = alpha + beta + gamma
    return csd.vertcat(x, y, theta)


def inverseKinematics(pose):
    """
    Planar IK for the 3-link arm.
    pose = [x_tip, y_tip, theta_tip]
    """
    xTip = pose[0]
    yTip = pose[1]
    thetaTip = pose[2]

    xJointBucket = xTip - C.lenLM * csd.cos(thetaTip)
    yJointBucket = yTip - C.lenLM * csd.sin(thetaTip)

    cosBeta = (
        xJointBucket**2
        + yJointBucket**2
        - C.lenBA**2
        - C.lenAL**2
    ) / (2 * C.lenBA * C.lenAL)
    # elbow-down configuration
    sinBeta = -csd.sqrt(1 - cosBeta**2)

    sinAlpha = (
        (C.lenBA + C.lenAL * cosBeta) * yJointBucket
        - C.lenAL * sinBeta * xJointBucket
    ) / (xJointBucket**2 + yJointBucket**2)
    cosAlpha = (
        (C.lenBA + C.lenAL * cosBeta) * xJointBucket
        + C.lenAL * sinBeta * yJointBucket
    ) / (xJointBucket**2 + yJointBucket**2)

    alpha = csd.atan2(sinAlpha, cosAlpha)
    beta = csd.atan2(sinBeta, cosBeta)
    gamma = thetaTip - alpha - beta

    return csd.vertcat(alpha, beta, gamma)


def inverseDynamics(q, qDot, qDDot, F):
    """
    Compute joint torques τ for given state (q, qDot, qDDot) and
    external tip force F (2D) expressed at the bucket frame.

    This builds M(q), b(q,qDot), g(q) using the CoM locations, masses and
    inertias from `excavatorConstants.py`, which have been adapted to
    the new URDF model0 (liftboom, tiltboom, scoop).
    """
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    alphaDot = qDot[0]
    betaDot = qDot[1]
    gammaDot = qDot[2]

    # Boom (liftboom)
    jacBoom = csd.vertcat(
        csd.horzcat(
            -C.lenBCoMBoom * csd.sin(alpha + C.angABCoMBoom), 0, 0
        ),
        csd.horzcat(
            C.lenBCoMBoom * csd.cos(alpha + C.angABCoMBoom), 0, 0
        ),
        csd.horzcat(1, 0, 0),
    )

    # Arm (tiltboom)
    jacArm = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.sin(alpha)
            - C.lenACoMArm
            * csd.sin(alpha + beta + C.angLACoMArm),
            -C.lenACoMArm
            * csd.sin(alpha + beta + C.angLACoMArm),
            0,
        ),
        csd.horzcat(
            C.lenBA * csd.cos(alpha)
            + C.lenACoMArm
            * csd.cos(alpha + beta + C.angLACoMArm),
            C.lenACoMArm
            * csd.cos(alpha + beta + C.angLACoMArm),
            0,
        ),
        csd.horzcat(1, 1, 0),
    )

    # Bucket (scoop)
    jacBucket = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.sin(alpha)
            - C.lenAL * csd.sin(alpha + beta)
            - C.lenLCoMBucket
            * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
            -C.lenAL * csd.sin(alpha + beta)
            - C.lenLCoMBucket
            * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
            -C.lenLCoMBucket
            * csd.sin(alpha + beta + gamma + C.angMLCoMBucket),
        ),
        csd.horzcat(
            C.lenBA * csd.cos(alpha)
            + C.lenAL * csd.cos(alpha + beta)
            + C.lenLCoMBucket
            * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
            C.lenAL * csd.cos(alpha + beta)
            + C.lenLCoMBucket
            * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
            C.lenLCoMBucket
            * csd.cos(alpha + beta + gamma + C.angMLCoMBucket),
        ),
        csd.horzcat(1, 1, 1),
    )

    # Time derivatives of Jacobians
    jacBoomDot = csd.vertcat(
        csd.horzcat(
            -C.lenBCoMBoom
            * csd.cos(alpha + C.angABCoMBoom)
            * alphaDot,
            0,
            0,
        ),
        csd.horzcat(
            -C.lenBCoMBoom
            * csd.sin(alpha + C.angABCoMBoom)
            * alphaDot,
            0,
            0,
        ),
        csd.horzcat(0, 0, 0),
    )

    jacArmDot = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.cos(alpha) * alphaDot
            - C.lenACoMArm
            * csd.cos(alpha + beta + C.angLACoMArm)
            * (alphaDot + betaDot),
            -C.lenACoMArm
            * csd.cos(alpha + beta + C.angLACoMArm)
            * (alphaDot + betaDot),
            0,
        ),
        csd.horzcat(
            -C.lenBA * csd.sin(alpha) * alphaDot
            - C.lenACoMArm
            * csd.sin(alpha + beta + C.angLACoMArm)
            * (alphaDot + betaDot),
            -C.lenACoMArm
            * csd.sin(alpha + beta + C.angLACoMArm)
            * (alphaDot + betaDot),
            0,
        ),
        csd.horzcat(0, 0, 0),
    )

    jacBucketDot = csd.vertcat(
        csd.horzcat(
            -C.lenBA * csd.cos(alpha) * alphaDot
            - C.lenAL
            * csd.cos(alpha + beta)
            * (alphaDot + betaDot)
            - C.lenLCoMBucket
            * csd.cos(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
            -C.lenAL
            * csd.cos(alpha + beta)
            * (alphaDot + betaDot)
            - C.lenLCoMBucket
            * csd.cos(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
            -C.lenLCoMBucket
            * csd.cos(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
        ),
        csd.horzcat(
            -C.lenBA * csd.sin(alpha) * alphaDot
            - C.lenAL
            * csd.sin(alpha + beta)
            * (alphaDot + betaDot)
            - C.lenLCoMBucket
            * csd.sin(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
            -C.lenAL
            * csd.sin(alpha + beta)
            * (alphaDot + betaDot)
            - C.lenLCoMBucket
            * csd.sin(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
            -C.lenLCoMBucket
            * csd.sin(
                alpha + beta + gamma + C.angMLCoMBucket
            )
            * (alphaDot + betaDot + gammaDot),
        ),
        csd.horzcat(0, 0, 0),
    )

    # Mass matrix
    M = (
        jacBoom[0:2, :].T @ C.massBoom @ jacBoom[0:2, :]
        + jacBoom[2, :].T @ C.moiBoom @ jacBoom[2, :]
        + jacArm[0:2, :].T @ C.massArm @ jacArm[0:2, :]
        + jacArm[2, :].T @ C.moiArm @ jacArm[2, :]
        + jacBucket[0:2, :].T @ C.massBucket @ jacBucket[0:2, :]
        + jacBucket[2, :].T @ C.moiBucket @ jacBucket[2, :]
    )

    # Coriolis/centrifugal term
    b = (
        jacBoom[0:2, :].T @ C.massBoom @ jacBoomDot[0:2, :] @ qDot
        + jacBoom[2, :].T @ (C.moiBoom @ jacBoomDot[2, :] @ qDot)
        + jacArm[0:2, :].T @ C.massArm @ jacArmDot[0:2, :] @ qDot
        + jacArm[2, :].T @ (C.moiArm @ jacArmDot[2, :] @ qDot)
        + jacBucket[0:2, :].T @ C.massBucket @ jacBucketDot[0:2, :] @ qDot
        + jacBucket[2, :].T @ (C.moiBucket @ jacBucketDot[2, :] @ qDot)
    )

    # Gravity
    g = (
        -jacBoom[0:2, :].T @ C.massBoom @ C.g
        - jacArm[0:2, :].T @ C.massArm @ C.g
        - jacBucket[0:2, :].T @ C.massBucket @ C.g
    )

    # External tip force mapped to joint space
    extF = jacBucket[0:2, :].T @ F

    return M @ qDDot + b + g - extF


def jointAngles(len):
    """
    Map actuator lengths -> joint angles.
    NOTE: linkage geometry (C.lenAF, lenHJ, ...) still comes from
    an approximate/legacy model. For the new URDF this is only a
    rough mapping; refine if you really need cylinder-level accuracy.
    """
    lenBoom = len[0]
    lenArm = len[1]
    lenBucket = len[2]

    R = C.lenBC
    theta = csd.atan2(C.iBC[1], C.iBC[0])
    alpha = (
        csd.acos(
            (-lenBoom**2 + C.lenBD**2 + C.lenBC**2)
            / (2 * R * C.lenBD)
        )
        + theta
        - C.angABD
    )

    R = C.lenAE
    theta = csd.atan2(C.bAE[0], C.bAE[1])
    beta = (
        csd.asin(
            (-lenArm**2 + C.lenAF**2 + C.lenAE**2)
            / (2 * R * C.lenAF)
        )
        - theta
        - C.angFAL
    )

    R = C.lenJG
    theta = csd.atan2(C.aJG[0], C.aJG[1])
    angLJH = (
        csd.asin(
            (-lenBucket**2 + C.lenHJ**2 + C.lenJG**2)
            / (2 * R * C.lenHJ)
        )
        - theta
    )

    R = csd.sqrt(
        (C.lenJL - C.lenHJ * csd.cos(angLJH)) ** 2
        + (C.lenHJ * csd.sin(angLJH)) ** 2
    )
    theta = csd.atan2(
        C.lenHJ * csd.sin(angLJH),
        C.lenJL - C.lenHJ * csd.cos(angLJH),
    )
    gamma = (
        csd.acos(
            (
                C.lenHK**2
                - C.lenJL**2
                - C.lenLK**2
                - C.lenHJ**2
                + 2 * C.lenJL * C.lenHJ * csd.cos(angLJH)
            )
            / (2 * R * C.lenLK)
        )
        - theta
        - C.angKLM
    )

    return csd.vertcat(alpha, beta, gamma)


def jointVel(q, actuatorVel):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    lenBoomDot = actuatorVel[0]
    lenArmDot = actuatorVel[1]
    lenBucketDot = actuatorVel[2]

    alphaDot = lenBoomDot / (
        0.0344 * alpha**3 - 0.1377 * alpha**2 - 0.0208 * alpha + 0.2956
    )
    betaDot = lenArmDot / (
        0.0312 * beta**3 + 0.2751 * beta**2 + 0.582 * beta + 0.0646
    )
    gammaDot = lenBucketDot / (
        0.0192 * gamma**3 + 0.0864 * gamma**2 + 0.045 * gamma - 0.1695
    )

    return csd.vertcat(alphaDot, betaDot, gammaDot)


def actuatorLen(q):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    # Polynomial fits from legacy system (kept as-is).
    lenBoom = (
        0.0086 * alpha**4
        - 0.0459 * alpha**3
        - 0.0104 * alpha**2
        + 0.2956 * alpha
        + 1.042
    )
    lenArm = (
        0.0078 * beta**4
        + 0.0917 * beta**3
        + 0.2910 * beta**2
        + 0.0646 * beta
        + 1.0149
    )
    lenBucket = (
        0.0048 * gamma**4
        + 0.0288 * gamma**3
        + 0.0225 * gamma**2
        - 0.1695 * gamma
        + 0.9434
    )

    return csd.vertcat(lenBoom, lenArm, lenBucket)


def actuatorVelFactor(q):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]

    velFactorBoom = (
        0.0344 * alpha**3 - 0.1377 * alpha**2 - 0.0208 * alpha + 0.2956
    )
    velFactorArm = (
        0.0312 * beta**3 + 0.2751 * beta**2 + 0.582 * beta + 0.0646
    )
    velFactorBucket = (
        0.0192 * gamma**3 + 0.0864 * gamma**2 + 0.045 * gamma - 0.1695
    )

    return csd.vertcat(velFactorBoom, velFactorArm, velFactorBucket)


def actuatorVel(q, qDot):
    alpha = q[0]
    beta = q[1]
    gamma = q[2]
    alphaDot = qDot[0]
    betaDot = qDot[1]
    gammaDot = qDot[2]

    lenBoomDot = alphaDot * (
        0.0344 * alpha**3 - 0.1377 * alpha**2 - 0.0208 * alpha + 0.2956
    )
    lenArmDot = betaDot * (
        0.0312 * beta**3 + 0.2751 * beta**2 + 0.582 * beta + 0.0646
    )
    lenBucketDot = gammaDot * (
        0.0192 * gamma**3 + 0.0864 * gamma**2 + 0.045 * gamma - 0.1695
    )

    return csd.vertcat(lenBoomDot, lenArmDot, lenBucketDot)


def motorVel(q, qDot):
    lenDot = actuatorVel(q, qDot)
    lenBoomDot = lenDot[0]
    lenArmDot = lenDot[1]
    lenBucketDot = lenDot[2]

    angVelMotorBoom = 2444.16 * lenBoomDot
    angVelMotorArm = 2444.16 * lenArmDot
    angVelMotorBucket = 2444.16 * lenBucketDot

    return csd.vertcat(angVelMotorBoom, angVelMotorArm, angVelMotorBucket)


def motorTorque(q, qDot, qDDot, F):
    T = inverseDynamics(q, qDot, qDDot, F)
    TBoom = T[0]
    TArm = T[1]
    TBucket = T[2]

    r = actuatorVelFactor(q)
    rBoom = r[0]
    rArm = r[1]
    rBucket = r[2]

    TMotorBoom = TBoom / (2444.16 * rBoom * 0.64)
    TMotorArm = TArm / (2444.16 * rArm * 0.64)
    TMotorBucket = TBucket / (2444.16 * rBucket * 0.64)

    return csd.vertcat(TMotorBoom, TMotorArm, TMotorBucket)


def motorTorqueLimit(motorVel, dutyCycle):
    match dutyCycle:
        case DutyCycle.S1:
            return (
                -1.4073e-7 * motorVel**3
                + 1.7961e-5 * motorVel**2
                - 0.0147 * motorVel
                + 19.9091
            )
        case DutyCycle.S2_60:
            return (
                -1.3568e-10 * motorVel**4
                - 1.4682e-7 * motorVel**3
                + 5.5744e-5 * motorVel**2
                - 0.0159 * motorVel
                + 25.0769
            )
        case DutyCycle.S2_30:
            return (
                -2.0433e-7 * motorVel**3
                + 5.1865e-5 * motorVel**2
                - 0.0105 * motorVel
                + 30.9119
            )
        case DutyCycle.PEAK:
            return (
                4.0269e-9 * motorVel**4
                - 3.7090e-6 * motorVel**3
                + 8.513e-4 * motorVel**2
                - 0.05787 * motorVel
                + 60.2614
            )
        case _:
            return (
                -1.4073e-7 * motorVel**3
                + 1.7961e-5 * motorVel**2
                - 0.0147 * motorVel
                + 19.9091
            )
