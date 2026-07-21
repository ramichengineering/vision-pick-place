## Variables:
for this bot, everything's a VECTOR LENGTH 7.

q=data.qpos[ARM] - Current joint angles in radians
q_des - target angles
qdot = data.qvel[ARM] - joint angular velocities in rad/sec. dq/dt
tau - the torque output. this is the only thing I'm controlling
qrfc_bias = data.qrfc_bias[ARM] - torque already acting from coriolis and gravity 
kp, kd - gain vectors

error: e= q_des - q

## The control law:
tau = kp * e - kd * qdot [+qrfc_bias]

each joint is pretty much obeying newton's second law for rotation
 - I * qddot = tau_applied
wher I is joint's inertia
qddot (double dot q) is angular accel, derivative of ang. velocity of course. 

## Mass spring damper equation
I * ë  +  kd * ė  +  kp * e  =  0
I: mass
eddot: error accel
edot: error vel

## Damping math
ζ = kd / (2 * sqrt(kp * I))

ζ < 1 (underdamped): overshoots and oscillates before settling. Too little kd.
ζ = 1 (critically damped): fastest possible settling with no overshoot. 
This is what I want
ζ > 1 (overdamped): no overshoot but sluggish, crawls to target. Too much kd.

Practical tuning loop:

Start with modest kp, kd near critical.
Too sluggish / big steady error → raise kp.
Oscillating / overshooting → raise kd (or lower kp).
Buzzing/unstable at the wrist → lower those joints' gains.

I remember this from my EE classes. 

## Writing PD controller:
-Want each joint to reach target angle q_des
-Proportional: kp*(q_des-q) Basically a spring. 
-Derivative: (-kd*qdot) damper

