import json
from pathlib import Path

import numpy as np
from mpmath import iv, mp


iv.dps = 80
mp.dps = 100
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "experiments" / "global_recovery"
stored = np.load(DATA / "lqr_matrices.npz")


def interval_float(value):
    """Contain the stored IEEE float and one adjacent float on either side."""
    value = float(value)
    lower = np.nextafter(value, -np.inf)
    upper = np.nextafter(value, np.inf)
    return iv.mpf([repr(float(lower)), repr(float(upper))])


def point(value):
    return iv.mpf([str(value), str(value)])


def lower(value):
    """Export an interval lower endpoint to binary64, rounded downward."""
    return float(np.nextafter(float(value.a), -np.inf))


def upper(value):
    """Export an interval upper endpoint to binary64, rounded upward."""
    return float(np.nextafter(float(value.b), np.inf))


def round_up(value):
    """Round a real-valued scalar export upward by one binary64 ulp."""
    return float(np.nextafter(float(value), np.inf))


def mp_to_upper(value):
    """Export a high-precision scalar as a conservative binary64 upper bound."""
    return round_up(float(value))


def sqrt_upper(value):
    return mp_to_upper(mp.sqrt(mp.mpf(repr(value))))


def abs_upper(value):
    return max(abs(lower(value)), abs(upper(value)))


def zeros(rows, cols):
    return [[point(0) for _ in range(cols)] for _ in range(rows)]


def eye(size):
    out = zeros(size, size)
    for i in range(size):
        out[i][i] = point(1)
    return out


def shape(matrix):
    return len(matrix), len(matrix[0])


def add(left_matrix, right_matrix):
    rows, cols = shape(left_matrix)
    return [
        [left_matrix[i][j] + right_matrix[i][j] for j in range(cols)]
        for i in range(rows)
    ]


def subtract(left_matrix, right_matrix):
    rows, cols = shape(left_matrix)
    return [
        [left_matrix[i][j] - right_matrix[i][j] for j in range(cols)]
        for i in range(rows)
    ]


def scale(matrix, scalar):
    rows, cols = shape(matrix)
    return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]


def transpose(matrix):
    rows, cols = shape(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def matmul(left_matrix, right_matrix):
    rows, inner = shape(left_matrix)
    inner_right, cols = shape(right_matrix)
    assert inner == inner_right
    out = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            total = point(0)
            for k in range(inner):
                total += left_matrix[i][k] * right_matrix[k][j]
            out[i][j] = total
    return out


def interval_norm_inf(matrix):
    row_bounds = []
    for row in matrix:
        total = 0.0
        for value in row:
            total = round_up(mp.mpf(repr(total)) + mp.mpf(repr(abs_upper(value))))
        row_bounds.append(total)
    return max(row_bounds)


def inflate(matrix, radius):
    uncertainty = iv.mpf([repr(-radius), repr(radius)])
    rows, cols = shape(matrix)
    return [
        [matrix[i][j] + uncertainty for j in range(cols)] for i in range(rows)
    ]


def matrix_series(A, B, horizon, order):
    """Enclose exp(Ah) and integral_0^h exp(As)B ds by Taylor series."""
    size, _ = shape(A)
    exponential = eye(size)
    exp_term = eye(size)
    for k in range(1, order + 1):
        exp_term = scale(matmul(exp_term, A), horizon / point(k))
        exponential = add(exponential, exp_term)

    integral = scale(B, horizon)
    int_term = integral
    for k in range(order):
        int_term = scale(matmul(A, int_term), horizon / point(k + 2))
        integral = add(integral, int_term)

    a_norm = interval_norm_inf(A)
    b_norm = interval_norm_inf(B)
    h_upper = upper(horizon)
    x_mp = mp.mpf(repr(a_norm)) * mp.mpf(repr(h_upper))
    exp_tail = mp_to_upper(
        mp.exp(x_mp) * x_mp ** (order + 1) / mp.factorial(order + 1)
    )
    int_tail = mp_to_upper(
        mp.mpf(repr(h_upper))
        * mp.mpf(repr(b_norm))
        * mp.exp(x_mp)
        * x_mp ** (order + 1)
        / mp.factorial(order + 2)
    )
    return inflate(exponential, exp_tail), inflate(integral, int_tail), {
        "order": order,
        "A_infinity_norm_upper": a_norm,
        "exp_tail_upper": exp_tail,
        "integral_tail_upper": int_tail,
    }


def interval_ldl_positive(matrix):
    """Interval LDL^T test; positive lower pivots certify positive definiteness."""
    size, _ = shape(matrix)
    symmetric = scale(add(matrix, transpose(matrix)), point("0.5"))
    L = zeros(size, size)
    pivots = []
    for i in range(size):
        L[i][i] = point(1)
    for k in range(size):
        pivot = symmetric[k][k]
        for j in range(k):
            pivot -= L[k][j] * L[k][j] * pivots[j]
        if lower(pivot) <= 0:
            return False, [lower(value) for value in pivots] + [lower(pivot)]
        pivots.append(pivot)
        for i in range(k + 1, size):
            numerator = symmetric[i][k]
            for j in range(k):
                numerator -= L[i][j] * L[k][j] * pivots[j]
            L[i][k] = numerator / pivot
    return True, [lower(value) for value in pivots]


P = [[interval_float(value) for value in row] for row in stored["P"]]
K = [[interval_float(value) for value in stored["K"].ravel()]]
h = interval_float(0.002)
gamma_endpoints = (interval_float(0.8), interval_float(1.25))
gamma_max = interval_float(1.25)
beta = point("0.0035")
radius = point("0.0016")
input_limit = point("0.5")

# Public plant constants, each enclosing the corresponding implementation float.
m2 = interval_float(0.6255677234174437)
l1 = interval_float(0.2)
r2 = interval_float(0.25569305436052964)
I1 = interval_float(0.031887199591513114)
I2 = interval_float(0.05086984812807257)
mass1 = interval_float(0.5234602302310271)
gravity = interval_float(9.81)
b = m2 * l1 * r2
d = I2
a = I1 + d + m2 * l1 * l1
Agrav = (mass1 * l1 + m2 * l1) * gravity
Bgrav = m2 * r2 * gravity

m00 = a + point(2) * b
m01 = d + b
det0 = m00 * d - m01 * m01
M0i = [[d / det0, -m01 / det0], [-m01 / det0, m00 / det0]]
Lq = [
    [Agrav + Bgrav, Bgrav],
    [Bgrav, Bgrav],
]
A0 = zeros(4, 4)
A0[0][2] = point(1)
A0[1][3] = point(1)
lower_left = matmul(M0i, Lq)
for i in range(2):
    for j in range(2):
        A0[i + 2][j] = lower_left[i][j]
B0 = [[point(0)], [point(0)], [M0i[0][1]], [M0i[1][1]]]

# Exact ZOH linear map F_gamma=exp(Ah)-gamma*Gamma*K.
E, Gamma, linear_tail = matrix_series(A0, B0, h, order=18)
linear_certificates = []
linear_ok = True
for gamma in gamma_endpoints:
    F = subtract(E, scale(matmul(Gamma, K), gamma))
    lhs = matmul(matmul(transpose(F), P), F)
    D = subtract(scale(P, point(1) - beta), lhs)
    passed, pivot_lowers = interval_ldl_positive(D)
    linear_ok = linear_ok and passed
    linear_certificates.append(
        {"gain": [lower(gamma), upper(gamma)], "ldl_pivot_lowers": pivot_lowers,
         "positive_definite": passed}
    )

# Verified inverse enclosure for P using a high-precision approximate inverse.
P_center = mp.matrix([[mp.mpf(float(value)) for value in row] for row in stored["P"]])
X_center = P_center ** -1
X = [[point(mp.nstr(X_center[i, j], 100)) for j in range(4)] for i in range(4)]
residual = subtract(eye(4), matmul(P, X))
residual_norm = interval_norm_inf(residual)
X_norm = interval_norm_inf(X)
inverse_error = mp_to_upper(
    mp.mpf(repr(X_norm)) * mp.mpf(repr(residual_norm))
    / (mp.mpf(1) - mp.mpf(repr(residual_norm)))
)
Pinv = [
    [X[i][j] + iv.mpf([-inverse_error, inverse_error]) for j in range(4)]
    for i in range(4)
]
inverse_ok = residual_norm < 1.0
diag_inverse_upper = [upper(Pinv[i][i]) for i in range(4)]
coordinate_envelope = [
    mp_to_upper(mp.mpf(repr(upper(radius))) * mp.sqrt(mp.mpf(repr(value))))
    for value in diag_inverse_upper
]

kpinvk = matmul(matmul(K, Pinv), transpose(K))[0][0]
input_upper = mp_to_upper(
    mp.mpf(repr(upper(gamma_max)))
    * mp.mpf(repr(upper(radius)))
    * mp.sqrt(mp.mpf(repr(upper(kpinvk))))
)
input_ok = input_upper <= lower(input_limit)

# A deliberately rounded-up one-hold state box.
box = [value*radius/point("0.0018") for value in
       [point("0.0011"), point("0.0012"), point("0.028"), point("0.056")]]
Q1, Q2, V1, V2 = box
c = iv.cos(iv.mpf([-upper(Q2), upper(Q2)]))
mass00 = a + point(2) * b * c
mass01 = d + b * c
det = mass00 * d - mass01 * mass01
Mi = [[d / det, -mass01 / det], [-mass01 / det, mass00 / det]]
Mi_abs = [[point(abs_upper(value)) for value in row] for row in Mi]
# M(q)^-1-M(0)^-1 = M(q)^-1 (M(0)-M(q)) M(0)^-1.
# Since |1-cos(q2)| <= q2^2/2, this bound scales quadratically
# on every box s*box, 0 <= s <= 1. Together with the cubic gravity
# and Coriolis remainders it certifies the entire ellipsoid, not just
# the outer radius used to construct the comparison box.
delta_M_abs = [[b*Q2**2, b*Q2**2/point(2)],
               [b*Q2**2/point(2), point(0)]]
M0i_abs = [[point(abs_upper(value)) for value in row] for row in M0i]
delta_Mi_abs = matmul(matmul(Mi_abs, delta_M_abs), M0i_abs)
effective_input = point(str(input_upper))
torque = [
    Agrav * Q1 + Bgrav * (Q1 + Q2) + b * Q2 * (point(2) * V1 * V2 + V2 * V2),
    Bgrav * (Q1 + Q2) + b * Q2 * V1 * V1 + effective_input,
]
acceleration = [
    Mi_abs[i][0] * torque[0] + Mi_abs[i][1] * torque[1] for i in range(2)
]
self_lhs = [
    upper(point(repr(coordinate_envelope[0])) + h * V1),
    upper(point(repr(coordinate_envelope[1])) + h * V2),
    upper(point(repr(coordinate_envelope[2])) + h * acceleration[0]),
    upper(point(repr(coordinate_envelope[3])) + h * acceleration[1]),
]
box_upper = [upper(value) for value in box]
box_ok = all(left <= right for left, right in zip(self_lhs, box_upper))

linear_torque = [
    (Agrav + Bgrav) * Q1 + Bgrav * Q2,
    Bgrav * (Q1 + Q2) + effective_input,
]
gravity_remainder = [
    Agrav * Q1**3 / point(6) + Bgrav * (Q1 + Q2) ** 3 / point(6),
    Bgrav * (Q1 + Q2) ** 3 / point(6),
]
coriolis = [
    b * Q2 * (point(2) * V1 * V2 + V2 * V2),
    b * Q2 * V1 * V1,
]
n_acceleration = []
for i in range(2):
    first = sum(delta_Mi_abs[i][j] * linear_torque[j] for j in range(2))
    second = sum(
        Mi_abs[i][j] * (gravity_remainder[j] + coriolis[j]) for j in range(2)
    )
    n_acceleration.append(first + second)

# Cancellation-free state-remainder kernel:
# |integral exp(A0 s) Bn n(s) ds| <= integral exp(|A0|s)|Bn| ds * n_bar.
Aabs = [[point(abs_upper(value)) for value in row] for row in A0]
Bn = [[point(0), point(0)], [point(0), point(0)],
      [point(1), point(0)], [point(0), point(1)]]
_, state_kernel, nonlinear_tail = matrix_series(Aabs, Bn, h, order=18)
state_remainder = [
    state_kernel[i][0] * n_acceleration[0]
    + state_kernel[i][1] * n_acceleration[1]
    for i in range(4)
]
Pabs = [[point(abs_upper(value)) for value in row] for row in P]
lyapunov_remainder_squared = point(0)
for i in range(4):
    for j in range(4):
        lyapunov_remainder_squared += (
            state_remainder[i] * Pabs[i][j] * state_remainder[j]
        )
lyapunov_remainder = iv.sqrt(lyapunov_remainder_squared)
ratio = (
    iv.sqrt(point(1) - beta) + lyapunov_remainder / radius
) ** 2
ratio_upper = upper(ratio)
ratio_threshold = lower(point(1) - beta / point(2))
nonlinear_ok = ratio_upper <= ratio_threshold

P_ok, P_pivots = interval_ldl_positive(P)
certificate_ok = all([P_ok, linear_ok, inverse_ok, input_ok, box_ok, nonlinear_ok])
result = {
    "certificate": certificate_ok,
    "arithmetic": (
        "mpmath interval arithmetic with outward-rounded input floats and "
        "outward-rounded binary64 scalar exports"
    ),
    "sample_period": [lower(h), upper(h)],
    "gain_interval": [lower(gamma_endpoints[0]), upper(gamma_endpoints[1])],
    "linear_beta": lower(beta),
    "linear_tail": linear_tail,
    "linear_endpoint_certificates": linear_certificates,
    "inverse_residual_infinity_norm_upper": residual_norm,
    "inverse_enclosure_error_infinity_norm": inverse_error,
    "P_norm_radius": upper(radius),
    "rho": mp_to_upper(mp.mpf(repr(upper(radius))) ** 2),
    "coordinate_envelope_upper": coordinate_envelope,
    "effective_input_upper": input_upper,
    "input_limit_lower": lower(input_limit),
    "one_hold_box_upper": box_upper,
    "one_hold_self_consistency_lhs": self_lhs,
    "nonlinear_acceleration_upper": [upper(value) for value in n_acceleration],
    "state_kernel_tail": nonlinear_tail,
    "state_remainder_upper": [upper(value) for value in state_remainder],
    "lyapunov_remainder_upper": upper(lyapunov_remainder),
    "V_ratio_upper": ratio_upper,
    "V_ratio_threshold_lower": ratio_threshold,
    "checks": {
        "P_positive_definite": P_ok,
        "linear_LMI": linear_ok,
        "inverse_enclosure": inverse_ok,
        "input_admissibility": input_ok,
        "one_hold_box": box_ok,
        "nonlinear_decrease": nonlinear_ok,
    },
    "P_ldl_pivot_lowers": P_pivots,
    "radial_scaling": "For e(0) in s*Omega^(1/2), s in [0,1], the one-hold box scales by s and the remainder by s^3 using the inverse-mass identity and trigonometric bounds.",
    "scope": (
        "Exact continuous-time Acrobot model with zero-order-held local LQR, "
        "stored P and K, gamma in [0.8,1.25], zero viscous drag, and exact state. "
        "Does not certify noisy estimation, the global rollout, the larger "
        "experimental switching rectangle, or the RK4 simulation map."
    ),
}
(ROOT / "interval_sampled_certificate.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
if not certificate_ok:
    raise SystemExit(1)
