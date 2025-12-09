import casadi as csd

"""
Geometry and inertial constants for the 3-DOF excavator arm used in the MPC model.

This version is adapted to the `model0` URDF:

    test_bench -> mountboom -> liftboom -> tiltboom -> scoop -> end_point

We treat the mechanism as a planar 3-link manipulator in the X–Z plane of the URDF.
In this file we only define the quantities that are actually used in
`excavatorModel.py` (all C.* references there).

2-D convention:
    x component  = URDF x
    y component  = URDF z
"""

# ---------------------------------------------------------------------------
# 1. Link-to-link geometry (from URDF)
# ---------------------------------------------------------------------------

# Boom (liftboom): vector from boom joint (mountboom->liftboom) to tilt joint
# in the boom frame, projected to X–Z.
bBA = csd.vertcat(0.20564, 0.420085)  # [m]

# Arm (tiltboom): vector from arm joint (liftboom->tiltboom) to bucket joint
aAL = csd.vertcat(0.25, 0.0)          # [m]

# Bucket (scoop): vector from bucket joint to bucket tip (end-effector) in
# the bucket frame.  Length is scaled from original big model
# (0.567 m on a 1.050 m arm) → same ratio on new 0.25 m arm.
lLM = csd.vertcat(0.135, 0.0)         # [m]  (≈ 0.54 * lenAL)

# ---------------------------------------------------------------------------
# 2. Link centre-of-mass locations (from URDF inertial origins, X–Z only)
# ---------------------------------------------------------------------------

# Boom (liftboom)
bCoMBoom = csd.vertcat(0.0623874, 0.2278545)     # [m]

# Arm (tiltboom)
aCoMArm = csd.vertcat(0.0931554, 0.0146127)      # [m]

# Bucket (scoop)
lCoMBucket = csd.vertcat(0.0290255, -0.0664655)  # [m]

# ---------------------------------------------------------------------------
# 3. Derived scalar lengths and angles used in the dynamics
# ---------------------------------------------------------------------------

# Link lengths
lenBA = csd.norm_2(bBA)       # boom joint-to-joint length
lenAL = csd.norm_2(aAL)       # arm joint-to-joint length
lenLM = csd.norm_2(lLM)       # bucket joint-to-tip length

# Distances joint -> CoM
lenBCoMBoom   = csd.norm_2(bCoMBoom)
lenACoMArm    = csd.norm_2(aCoMArm)
lenLCoMBucket = csd.norm_2(lCoMBucket)

# Angles from joint axis to CoM vectors
angABCoMBoom   = csd.atan2(bCoMBoom[1], bCoMBoom[0])
angLACoMArm    = csd.atan2(aCoMArm[1], aCoMArm[0])
angMLCoMBucket = csd.atan2(lCoMBucket[1], lCoMBucket[0])

# ---------------------------------------------------------------------------
# 4. Inertial parameters (from URDF inertias, iyy terms)
# ---------------------------------------------------------------------------

# Scalar masses [kg]
_massBoom   = 1.3268
_massArm    = 0.7769
_massBucket = 0.5724

# 2x2 translational mass matrices used in M(q)
massBoom   = _massBoom   * csd.DM.eye(2)
massArm    = _massArm    * csd.DM.eye(2)
massBucket = _massBucket * csd.DM.eye(2)

# Rotational inertias about the out-of-plane axis [kg m^2]
moiBoom   = csd.DM([[0.0277166]])
moiArm    = csd.DM([[0.007241]])
moiBucket = csd.DM([[0.0012744]])

# ---------------------------------------------------------------------------
# 5. Additional geometric quantities used in helper functions
#    (actuator / linkage kinematics).
#
# ⚠️ 这些目前是占位值，用来保证原来的辅助函数还能编译。
# 如果你将来要精确建模液压缸 / 连杆，请按新机械臂重新标定。
# ---------------------------------------------------------------------------

# Base-to-boom auxiliary vector (kept from original model, 2-D projection)
iBC = csd.vertcat(0.135, -0.264)
lenBC = csd.norm_2(iBC)

# Auxiliary vector on boom (used together with lenAE etc.)
# NOTE: placeholder; not taken from URDF.
bAE = csd.vertcat(-0.3, 0.25)
lenAE = csd.norm_2(bAE)

# One auxiliary vector on the arm for linkage geometry (placeholder)
aJG = csd.vertcat(0.15, 0.05)
lenJG = csd.norm_2(aJG)

# Scalar linkage lengths used in actuator kinematics (placeholders)
lenAF = csd.DM(0.25)
lenHJ = csd.DM(0.25)
lenHK = csd.DM(0.40)
lenJL = csd.DM(0.35)
lenLK = csd.DM(0.30)
lenBD = csd.DM(0.30)   # used in angle ABD

# Angle constants for linkage triangles (placeholders – keep inside (0, pi))
angABD = csd.DM(0.3)    # boom linkage triangle A-B-D
angFAL = csd.DM(0.4)    # arm linkage triangle F-A-L

# For the bucket linkage K-L-M we tie the angle to lenLK, lenLM, lenKM
lenKM = csd.DM(0.25)
angKLM = csd.acos((lenLM**2 + lenLK**2 - lenKM**2) / (2 * lenLK * lenLM))

# ---------------------------------------------------------------------------
# 6. Environment
# ---------------------------------------------------------------------------

# Ground height in the 2-D model (you can tune this to match your world frame)
yGround = csd.DM(0.0)

# Gravity vector [m/s^2]
g = csd.vertcat(0.0, -9.81)
