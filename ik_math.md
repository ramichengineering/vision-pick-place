# The Math Behind the Damped Least-Squares IK Solver

This solver answers an inverse-kinematics question: *what joint angles $q$ put the gripper's tool-center-point (TCP) at a target pose?* It works by repeatedly **linearizing** the forward kinematics and taking a **damped least-squares** step toward the goal — a Gauss–Newton / Levenberg–Marquardt scheme.

---

## 1. Forward kinematics and the TCP

Let $q \in \mathbb{R}^n$ be the joint angles ($n = 7$ arm joints on the Panda). Forward kinematics is a nonlinear map

$$
q \;\longmapsto\; \big(p(q),\, R(q)\big),
$$

giving the hand body's world position $p(q) \in \mathbb{R}^3$ and orientation $R(q) \in SO(3)$.

The point we actually control is not the hand origin but the TCP, a fixed offset $r = [0,0,0.10]^\top$ (10 cm) in the hand's **local** frame. Transforming that offset into world coordinates:

$$
p_{\text{tcp}}(q) \;=\; p(q) \;+\; R(q)\, r .
$$

Because $r$ is a pure translation, the TCP's orientation equals the hand's orientation, represented as a unit quaternion $\hat q_{\text{tcp}} \in \mathbb{H}$, $\lVert \hat q_{\text{tcp}}\rVert = 1$.

In code this is the `tcp_pose` computation: `R = xmat.reshape(3,3)`, `pos = xpos + R @ TCP_OFFSET`, and the orientation is `xquat`.

---

## 2. Task-space error

Each iteration measures a 6-vector error $e = [\,e_p;\, e_o\,] \in \mathbb{R}^6$ between the current TCP pose and the target.

### 2.1 Position error

Simple vector difference in world coordinates:

$$
e_p \;=\; p^{\text{target}} \;-\; p_{\text{tcp}}(q) \;\in\; \mathbb{R}^3 .
$$

### 2.2 Orientation error (the quaternion part)

Orientation error can't be a subtraction — orientations live on the curved manifold $SO(3)$, not a vector space. The standard trick is to find the **rotation that carries the current orientation onto the target**, then express it as a rotation vector.

With quaternions in the MuJoCo convention $\hat q = [w, x, y, z]$:

**Step 1 — inverse of the current orientation.** For a *unit* quaternion the conjugate is the inverse:

$$
\hat q_{\text{cur}}^{-1} \;=\; \overline{\hat q_{\text{cur}}} \;=\; [\,w,\, -x,\, -y,\, -z\,].
$$

This is `mju_negQuat`.

**Step 2 — the error quaternion.** Compose target with the inverse of current (Hamilton product $\otimes$):

$$
\hat q_{\text{err}} \;=\; \hat q_{\text{target}} \otimes \hat q_{\text{cur}}^{-1}.
$$

This is `mju_mulQuat`. $\hat q_{\text{err}}$ is the world-frame rotation from current to target. If they already match, $\hat q_{\text{err}} = [1,0,0,0]$ (identity).

**Step 3 — quaternion to rotation vector.** A unit quaternion encodes an axis $\hat u$ and angle $\theta$ as $\hat q_{\text{err}} = [\cos\tfrac{\theta}{2},\ \hat u \sin\tfrac{\theta}{2}]$. Extract the axis·angle **rotation vector**:

$$
e_o \;=\; \hat u\,\theta,
\qquad
\theta \;=\; 2\,\operatorname{atan2}\!\big(\lVert \mathbf{v}\rVert,\, w\big),
\qquad
\hat u \;=\; \frac{\mathbf{v}}{\lVert \mathbf{v}\rVert},
$$

where $\mathbf{v} = [x,y,z]$ is the vector part of $\hat q_{\text{err}}$. If $\theta > \pi$ it is wrapped to $\theta - 2\pi$ so the arm rotates the *short* way. This is `mju_quat2Vel` (with $\mathrm{d}t = 1$). The result $e_o \in \mathbb{R}^3$ is a genuine vector — small, additive, and directly usable by the Jacobian.

Convergence is declared when $\lVert e_p \rVert < \texttt{pos\_tol}$ and $\lVert e_o \rVert < \texttt{ori\_tol}$.

---

## 3. The Jacobian

The Jacobian linearizes forward kinematics: it relates joint velocities $\dot q$ to the TCP's spatial velocity (linear stacked over angular):

$$
\begin{bmatrix} v \\ \omega \end{bmatrix}
\;=\;
\underbrace{\begin{bmatrix} J_p \\ J_r \end{bmatrix}}_{J}\, \dot q,
\qquad
J \in \mathbb{R}^{6 \times n}.
$$

- $J_p \in \mathbb{R}^{3\times n}$ (`_jacp`): how each joint moves the TCP **point** in world translation.
- $J_r \in \mathbb{R}^{3\times n}$ (`_jacr`): how each joint rotates the hand body (angular velocity).

MuJoCo's `mj_jac` returns this **geometric Jacobian** for the specified point on the specified body. Only the first $n$ arm columns are kept (`[:, :ARM_DOF]`), discarding the finger joints. For a position-only solve, just $J_p$ and $e_p$ are used.

The first-order model each iteration is therefore

$$
J\,\Delta q \;\approx\; e .
$$

---

## 4. Damped least squares (the update rule)

We want $\Delta q$ solving $J\,\Delta q = e$, but $J$ is generally **not invertible**: it is non-square ($6 \times 7$, redundant) and becomes **rank-deficient near singularities** — configurations where the arm loses a direction of instantaneous motion.

### 4.1 Why not the plain pseudo-inverse

The minimum-norm least-squares solution is

$$
\Delta q \;=\; J^{+} e \;=\; J^\top \big(J J^\top\big)^{-1} e .
$$

Near a singularity $J J^\top$ becomes ill-conditioned; its small singular values invert into huge factors, producing enormous, unstable joint velocities.

### 4.2 The damping fix (Levenberg–Marquardt)

Add a regularizer $\lambda^2 I$ before inverting:

$$
\boxed{\;\Delta q \;=\; J^\top \big(J J^\top + \lambda^2 I\big)^{-1} e\;}
$$

with $\lambda = \texttt{damping}$ (here $10^{-2}$, so $\lambda^2 = 10^{-4}$). Equivalently, this minimizes a *regularized* objective

$$
\Delta q \;=\; \arg\min_{\Delta q}\ \lVert J\,\Delta q - e\rVert^2 \;+\; \lambda^2 \lVert \Delta q\rVert^2 ,
$$

which explicitly penalizes large steps. Effects:

- **Away from singularities:** behaves like the true pseudo-inverse (accurate).
- **Near singularities:** trades a little tracking accuracy for a **bounded, stable** step.

In SVD terms, a singular value $\sigma_i$ is inverted not as $1/\sigma_i$ but as $\sigma_i / (\sigma_i^2 + \lambda^2)$, which stays finite as $\sigma_i \to 0$.

### 4.3 Why the $J J^\top$ form

The identity

$$
J^\top \big(J J^\top + \lambda^2 I\big)^{-1} \;=\; \big(J^\top J + \lambda^2 I\big)^{-1} J^\top
$$

holds, but the two differ in the size of the linear system solved. The code uses the left form because $J J^\top$ is **task-dimensional** ($6\times6$ or $3\times3$) — a smaller solve than the $7\times7$ $J^\top J$.

---

## 5. Stepping, damping the step, and limits

The raw $\Delta q$ is applied with a scalar gain $\alpha = \texttt{step}$ (here $0.5$):

$$
q \;\leftarrow\; q \;+\; \alpha\,\Delta q .
$$

Because the linear model $J\Delta q \approx e$ is only valid locally, taking a **fraction** of the step keeps the nonlinear iteration from overshooting — a fixed line-search factor. Empirically the error then decays by roughly $\alpha$ per iteration.

After each step, every limited joint is clamped back into its range:

$$
q_j \;\leftarrow\; \operatorname{clip}(q_j,\ q_j^{\min},\ q_j^{\max}).
$$

Iterate until the tolerances are met or `max_iters` is reached.

### Redundancy and the null space

With $n = 7$ joints and 6 task constraints, the solution has a 1-dimensional **null space**: joint motions $\Delta q_{\text{null}}$ with $J\,\Delta q_{\text{null}} = 0$ that reconfigure the arm **without moving the TCP**. The damped pseudo-inverse silently selects the *minimum-norm* update, so two runs from different seeds can converge to different valid configurations — the "IK may find another" case.

---

## 6. Algorithm summary

$$
\begin{aligned}
&\textbf{repeat:} \\
&\quad p_{\text{tcp}}, \hat q_{\text{tcp}} \leftarrow \text{FK}(q) &&\text{(mj\_kinematics, tcp\_pose)}\\
&\quad e_p = p^{\text{target}} - p_{\text{tcp}} \\
&\quad e_o = \operatorname{quat2vel}\!\big(\hat q_{\text{target}} \otimes \hat q_{\text{tcp}}^{-1}\big) \\
&\quad \textbf{if } \lVert e_p\rVert<\text{tol}_p \text{ and } \lVert e_o\rVert<\text{tol}_o:\ \textbf{return } q\ \text{(success)} \\
&\quad J = [\,J_p;\, J_r\,] \leftarrow \text{mj\_jac} \\
&\quad \Delta q = J^\top (J J^\top + \lambda^2 I)^{-1} e \\
&\quad q \leftarrow \operatorname{clip}\big(q + \alpha\,\Delta q\big)
\end{aligned}
$$

---

## 7. Worked numerical example (one iteration)

Traced on a 3-link planar stand-in (same helper math, real geometric Jacobian). Seed $q=[0.3,0.5,0.4]$, target the pose of $[0.6,0.7,0.2]$.

**Position error.** $p^{\text{target}} - p_{\text{tcp}} = [0.342,0.658,0] - [0.568,0.490,0] = [-0.226, 0.168, 0]$, magnitude $281.5\,\text{mm}$.

**Orientation error.** Current $\hat q = [0.825,0,0,0.565]$.
- $\hat q^{-1} = [0.825,0,0,-0.565]$
- $\hat q_{\text{err}} = \hat q_{\text{target}}\otimes\hat q^{-1} = [0.989,0,0,0.149]$
- $e_o = \operatorname{quat2vel} = [0,0,0.30]$ — i.e. $17.19^\circ$ about $+z$, matching the $85.94^\circ - 68.75^\circ$ heading gap.

**Jacobian** ($6\times3$):

$$
J=\begin{bmatrix}
-0.490 & -0.402 & -0.186\\
\;\;0.568 & \;\;0.282 & \;\;0.073\\
0&0&0\\ 0&0&0\\ 0&0&0\\
1&1&1
\end{bmatrix}
$$

The three zero rows are the unconstrained directions (out-of-plane translation and tilt) — exactly what $\lambda^2 I$ keeps invertible.

**DLS step.** With $\lambda^2 = 10^{-4}$: $\Delta q = [-0.085, 0.904, -0.519]$. Applying $\alpha=0.5$: $q \leftarrow [0.257, 0.952, 0.141]$ (no limits hit).

**Convergence.** Error roughly halves each iteration:

| iter | pos err (mm) | ori err (deg) |
|-----:|-------------:|--------------:|
| 0 | 281.49 | 17.19 |
| 1 | 144.20 | 8.59 |
| 2 | 71.82 | 4.29 |
| 3 | 35.77 | 2.15 |
| 4 | 17.88 | 1.07 |
| 5 | 8.96 | 0.54 |
| 9 | 0.57 | 0.034 → **converged** |

The final joints recover a valid solution to the target pose.

---

## 8. Variable glossary

Every quantity that appears above, grouped by role. Dimensions are for the Panda ($n = 7$ arm joints; full model has 9 DOF including two fingers).

### Configuration and pose

| Symbol | Code | Type / size | What it represents |
|---|---|---|---|
| $q$ | `d.qpos[ARM]` | $\mathbb{R}^n$ (7) | **Joint angles** — the unknowns being solved for. `ARM` indexes the arm joints inside the full 9-DOF `qpos`. |
| $q_{\text{init}}$ | `q_init` | $\mathbb{R}^n$ | **Initial guess** the iteration starts from. A good seed makes convergence fast and biases *which* solution is found. |
| $r$ | `TCP_OFFSET` | $\mathbb{R}^3$ | **Tool offset** $[0,0,0.10]$ — the grasp point's location in the hand's *local* frame, 10 cm along local $+z$. Constant. |
| $p(q)$ | `d.xpos[hand_id]` | $\mathbb{R}^3$ | **Hand-body origin** position in world coordinates, from forward kinematics. |
| $R(q)$ | `d.xmat[hand_id]` | $3\times3$ | **Hand orientation** as a rotation matrix (world $\leftarrow$ local). Used to rotate $r$ into world axes. |
| $p_{\text{tcp}}$ | `tcp_pos` | $\mathbb{R}^3$ | **Current TCP position** in world = $p(q) + R(q)\,r$. |
| $\hat q_{\text{tcp}}$ | `tcp_quat` | $\mathbb{H}$ (4) | **Current TCP orientation** as a unit quaternion $[w,x,y,z]$; equals the hand's `xquat` since $r$ is a pure translation. |

### Targets and errors

| Symbol | Code | Type / size | What it represents |
|---|---|---|---|
| $p^{\text{target}}$ | `target_pos` | $\mathbb{R}^3$ | **Desired TCP position** in world coordinates (the goal). |
| $\hat q_{\text{target}}$ | `target_quat` | $\mathbb{H}$ (4) | **Desired TCP orientation** quaternion. If `None`, orientation is ignored and only position is solved. |
| $e$ | `err` | $\mathbb{R}^6$ | **Full task-space error** stacked as $[e_p;\,e_o]$. |
| $e_p$ | `err[:3]` | $\mathbb{R}^3$ | **Position error** $= p^{\text{target}} - p_{\text{tcp}}$. |
| $e_o$ | `err[3:]` | $\mathbb{R}^3$ | **Orientation error** as a rotation vector (axis·angle). Zero when orientation is not being solved. |
| $\hat q_{\text{cur}}^{-1}$ | `neg` | $\mathbb{H}$ (4) | **Inverse of current orientation** (quaternion conjugate) — the "undo current rotation" step. |
| $\hat q_{\text{err}}$ | `qerr` | $\mathbb{H}$ (4) | **Error quaternion** $= \hat q_{\text{target}} \otimes \hat q_{\text{cur}}^{-1}$; the rotation from current to target. |
| $\hat u,\ \theta$ | (inside `mju_quat2Vel`) | axis $\in\mathbb{R}^3$, angle $\in\mathbb{R}$ | **Axis and angle** decomposition of $\hat q_{\text{err}}$; combine to $e_o = \hat u\,\theta$. |
| $\lVert e_p\rVert,\ \lVert e_o\rVert$ | `pos_err`, `ori_err` | scalars | **Error magnitudes** (m, rad) checked against tolerances and reported. |

### Jacobian and the update

| Symbol | Code | Type / size | What it represents |
|---|---|---|---|
| $J_p$ | `_jacp` | $3\times n$ | **Translational Jacobian** — how each joint's velocity moves the TCP point linearly in world. |
| $J_r$ | `_jacr` | $3\times n$ | **Rotational Jacobian** — how each joint's velocity rotates the hand (angular velocity). |
| $J$ | `J` | $6\times n$ (or $3\times n$) | **Stacked Jacobian** $[J_p; J_r]$; only the first `ARM_DOF` columns are kept. Linearizes FK: $J\,\Delta q \approx e$. |
| $J J^\top$ | `JJt` | $6\times6$ (or $3\times3$) | **Gram matrix** inverted in the DLS step; task-dimensional (small). |
| $\lambda$ | `damping` | scalar ($10^{-2}$) | **Damping factor** that regularizes the inverse near singularities. |
| $\lambda^2$ | `lam2` | scalar ($10^{-4}$) | **Squared damping** actually added to the diagonal, $J J^\top + \lambda^2 I$. |
| $\Delta q$ | `dq` | $\mathbb{R}^n$ | **Raw joint update** from the damped least-squares solve. |
| $\alpha$ | `step` | scalar ($0.5$) | **Step gain** — fraction of $\Delta q$ actually applied, keeping the nonlinear step stable. |
| $q_j^{\min}, q_j^{\max}$ | `m.jnt_range[j]` | scalars | **Joint limits** used to clip each updated joint back into its valid range. |

### Solver parameters and outputs

| Symbol | Code | Value | What it represents |
|---|---|---|---|
| — | `pos_tol` | $10^{-3}$ m | **Position tolerance** for declaring success (1 mm). |
| — | `ori_tol` | $10^{-2}$ rad | **Orientation tolerance** for success (~0.57°). |
| — | `max_iters` | 200 | **Iteration cap** before giving up. |
| — | `use_ori` | bool | Whether a `target_quat` was supplied (6-DOF vs position-only solve). |
| — | `IKResult.q` | $\mathbb{R}^9$ | **Full solution** `qpos` (all 9 joints, fingers included). |
| — | `IKResult.success` | bool | Whether tolerances were met within `max_iters`. |
| — | `IKResult.iters` | int | Iterations actually taken. |
| — | `IKResult.pos_err`, `.ori_err` | scalars | **Final errors** (m, rad). |

### MuJoCo helper calls

| Call | Role |
|---|---|
| `mj_name2id` | Look up the integer id of the `"hand"` body. |
| `mj_kinematics` | Forward kinematics only (body poses from `qpos`) — cheap, no dynamics. |
| `mj_comPos` | Compute center-of-mass quantities that `mj_jac` requires. |
| `mj_jac` | Geometric Jacobian of a given point on a given body → fills `_jacp`, `_jacr`. |
| `mju_negQuat` | Quaternion conjugate/inverse → `neg`. |
| `mju_mulQuat` | Quaternion (Hamilton) product → `qerr`. |
| `mju_quat2Vel` | Quaternion → rotation-vector angular error → `err[3:]`. |
