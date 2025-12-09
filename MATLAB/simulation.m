addpath(genpath(pwd))

e = excavatorModel(excavatorConstants, 0, 0, 0);
%{
Boom (0.905 - 1.305 m) (-0.4737 - 1.0923 rad)
Arm (1.046 - 1.556 m) (-2.5848 - -0.5103 rad)
Bucket (0.84 - 1.26 m) (-2.8659 - 0.7839 rad)
%}

[alpha, beta, gamma] = e.setLengths(1.2, 1.2, 1.2)

%[lenBoom, lenArm, lenBucket] = e.setAngles(0.5, -1, -1)

%[TBoom, TArm, TBucket] = e.inverseDynamics(1,2,3,4,5,6,[0;-1000])

%[FBoom, FArm, FBucket] = e.calcForces(3000, 2000, 1000)

e.visualise;
