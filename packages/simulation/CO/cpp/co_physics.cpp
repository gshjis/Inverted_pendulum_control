#include "co_physics.hpp"

#include <cmath>
#include <random>

StateDot3 compute_ddq(const State3& q, const StateDot3& dq, double F_total,
                       const PlantParams& p, bool single_mode) {
    const double x = q.x;
    const double th1 = q.theta1;
    const double th2 = q.theta2;
    const double dx = dq.x_dot;
    const double dth1 = dq.theta1_dot;
    const double dth2 = dq.theta2_dot;

    const double c1 = std::cos(th1);
    const double s1 = std::sin(th1);
    const double c12 = std::cos(th1 + th2);
    const double s12 = std::sin(th1 + th2);
    const double c2 = std::cos(th2);
    const double s2 = std::sin(th2);

    const double A = p.m1 * p.L1 + p.m2 * p.l1;
    const double B = p.m2 * p.L2;

    const double M11 = p.M + p.m1 + p.m2;
    const double M12 = A * c1 + B * c12;
    const double M13 = B * c12;
    const double M22 = p.J1 + p.m1 * p.L1 * p.L1 + p.J2 + p.m2 * (p.l1 * p.l1 + p.L2 * p.L2 + 2.0 * p.l1 * p.L2 * c2);
    const double M23 = p.J2 + p.m2 * p.L2 * p.L2 + p.m2 * p.l1 * p.L2 * c2;
    const double M33 = p.J2 + p.m2 * p.L2 * p.L2;

    const double K = A * dth1 * s1 + B * (dth1 + dth2) * s12;
    const double C12 = -K;
    const double C13 = -B * (dth1 + dth2) * s12;
    const double C22 = -p.m2 * p.l1 * p.L2 * s2 * dth2;
    const double C23 = -p.m2 * p.l1 * p.L2 * s2 * (dth1 + dth2);
    const double C32 = p.m2 * p.l1 * p.L2 * s2 * dth1;

    const double G2 = -A * p.g * s1 - B * p.g * s12;
    const double G3 = -B * p.g * s12;

    const double rhs1 = (F_total - p.b_c * dx) - C12 * dth1 - C13 * dth2;
    const double rhs2 = (-p.b_1 * dth1) - C22 * dth1 - C23 * dth2 - G2;
    const double rhs3 = (-p.b_2 * dth2) - C32 * dth1 - G3;

    if (single_mode) {
        const double det = M11 * M22 - M12 * M12;
        if (std::abs(det) > 1e-15) {
            const double ddx = (rhs1 * M22 - M12 * rhs2) / det;
            const double dth1_2 = (M11 * rhs2 - M12 * rhs1) / det;
            return StateDot3{ddx, dth1_2, 0.0};
        }
        return StateDot3{0.0, 0.0, 0.0};
    }

    // Solve 3x3: M * t = rhs using Cramer's rule for simplicity.
    const double a11 = M11, a12 = M12, a13 = M13;
    const double a21 = M12, a22 = M22, a23 = M23;
    const double a31 = M13, a32 = M23, a33 = M33;

    const double b1 = rhs1, b2 = rhs2, b3 = rhs3;

    const double detM = a11 * (a22 * a33 - a23 * a32)
                       - a12 * (a21 * a33 - a23 * a31)
                       + a13 * (a21 * a32 - a22 * a31);
    if (std::abs(detM) < 1e-18) {
        return StateDot3{0.0, 0.0, 0.0};
    }

    auto det3 = [](double p11, double p12, double p13,
                    double p21, double p22, double p23,
                    double p31, double p32, double p33) {
        return p11 * (p22 * p33 - p23 * p32)
             - p12 * (p21 * p33 - p23 * p31)
             + p13 * (p21 * p32 - p22 * p31);
    };

    const double det1 = det3(b1, a12, a13, b2, a22, a23, b3, a32, a33);
    const double det2 = det3(a11, b1, a13, a21, b2, a23, a31, b3, a33);
    const double det3v = det3(a11, a12, b1, a21, a22, b2, a31, a32, b3);

    return StateDot3{det1 / detM, det2 / detM, det3v / detM};
}

double sample_noise_force(const NoiseForceCPP& n) {
    if (n.std == 0.0) {
        return n.mean;
    }
    static thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(n.mean, n.std);
    return dist(rng);
}

void rk4_step(State3& q, StateDot3& dq, double F_total, double dt,
              const PlantParams& p, bool single_mode) {
    auto add_q = [](const State3& a, const StateDot3& k, double scale) {
        return State3{a.x + scale * k.x_dot,
                       a.theta1 + scale * k.theta1_dot,
                       a.theta2 + scale * k.theta2_dot};
    };

    auto add_dq = [](const StateDot3& a, const StateDot3& k, double scale) {
        return StateDot3{a.x_dot + scale * k.x_dot,
                          a.theta1_dot + scale * k.theta1_dot,
                          a.theta2_dot + scale * k.theta2_dot};
    };

    const StateDot3 k1 = compute_ddq(q, dq, F_total, p, single_mode);
    const State3 q2 = add_q(q, dq, 0.5 * dt);
    const StateDot3 dq2 = add_dq(dq, k1, 0.5 * dt);

    const StateDot3 k2 = compute_ddq(q2, dq2, F_total, p, single_mode);
    const State3 q3 = add_q(q, dq2, 0.5 * dt);
    const StateDot3 dq3 = add_dq(dq, k2, 0.5 * dt);

    const StateDot3 k3 = compute_ddq(q3, dq3, F_total, p, single_mode);
    const State3 q4 = add_q(q, dq3, dt);
    const StateDot3 dq4 = add_dq(dq, k3, dt);

    const StateDot3 k4 = compute_ddq(q4, dq4, F_total, p, single_mode);

    // Update dq: dq += dt/6 * (k1 + 2k2 + 2k3 + k4)
    dq.x_dot     += (dt / 6.0) * (k1.x_dot + 2.0 * k2.x_dot + 2.0 * k3.x_dot + k4.x_dot);
    dq.theta1_dot += (dt / 6.0) * (k1.theta1_dot + 2.0 * k2.theta1_dot + 2.0 * k3.theta1_dot + k4.theta1_dot);
    dq.theta2_dot += (dt / 6.0) * (k1.theta2_dot + 2.0 * k2.theta2_dot + 2.0 * k3.theta2_dot + k4.theta2_dot);

    // Update q: q += dt/6 * (dq + 2*dq2 + 2*dq3 + dq4)
    q.x      += (dt / 6.0) * (dq.x_dot + 2.0 * dq2.x_dot + 2.0 * dq3.x_dot + dq4.x_dot);
    q.theta1 += (dt / 6.0) * (dq.theta1_dot + 2.0 * dq2.theta1_dot + 2.0 * dq3.theta1_dot + dq4.theta1_dot);
    q.theta2 += (dt / 6.0) * (dq.theta2_dot + 2.0 * dq2.theta2_dot + 2.0 * dq3.theta2_dot + dq4.theta2_dot);

    // Normalize pendulum angles to [0, 2π) to prevent floating-point drift
    // and ensure consistent error computation in the controller.
    constexpr double TWO_PI = 2.0 * M_PI;
    q.theta1 = std::fmod(q.theta1, TWO_PI);
    if (q.theta1 < 0.0) q.theta1 += TWO_PI;
    q.theta2 = std::fmod(q.theta2, TWO_PI);
    if (q.theta2 < 0.0) q.theta2 += TWO_PI;
}
