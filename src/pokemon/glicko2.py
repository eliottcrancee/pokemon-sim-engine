import math
from numba import njit


@njit
def calculate_new_rating(r, rd, period_results):
    """Calculates the new rating and rating deviation based on period results.

    period_results: list of tuples (opponent_r, opponent_rd, score)
    score = 1 for win, 0.5 for draw, 0 for loss.

    Returns: (new_r, new_rd)
    """
    mu = (r - 1500) / 173.7178
    phi = rd / 173.7178

    # Step 1: variance v
    v_inv = 0.0
    for i in range(len(period_results)):
        r_j = period_results[i][0]
        rd_j = period_results[i][1]

        mu_j = (r_j - 1500) / 173.7178
        phi_j = rd_j / 173.7178

        g_phi = g(phi_j)
        e_val = E(mu, mu_j, phi_j)

        v_inv += (g_phi**2) * e_val * (1 - e_val)

    # Avoid division by zero if no games or variance is zero
    if v_inv == 0:
        return r, rd

    v = 1 / v_inv

    # Step 2: update φ and μ
    sum_outcomes = 0.0
    for i in range(len(period_results)):
        r_j = period_results[i][0]
        rd_j = period_results[i][1]
        s_j = period_results[i][2]

        mu_j = (r_j - 1500) / 173.7178
        phi_j = rd_j / 173.7178

        g_phi = g(phi_j)
        e_val = E(mu, mu_j, phi_j)

        sum_outcomes += g_phi * (s_j - e_val)

    phi_prime = 1 / math.sqrt(1 / phi**2 + 1 / v)
    mu_prime = mu + phi_prime**2 * sum_outcomes

    # Step 3: back to rating scale
    new_r = 173.7178 * mu_prime + 1500
    new_rd = 173.7178 * phi_prime

    return new_r, new_rd


# --- Glicko-2 helper functions ---
@njit
def g(phi):
    return 1 / math.sqrt(1 + 3 * (phi**2) / (math.pi**2))


@njit
def E(mu, mu_j, phi_j):
    return 1 / (1 + math.exp(-g(phi_j) * (mu - mu_j)))
