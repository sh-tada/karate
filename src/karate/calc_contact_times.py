from karate.elements_to_orbit import true_anomaly_to_eccentric_anomaly
from karate.elements_to_orbit import t0_to_tperi
from karate.elements_to_orbit import eccentric_anomaly_to_t_from_tperi
import jax.numpy as jnp
import jax
from jax import jit


@jit
def calc_contact_times_circular(rp_over_rs, period, a_over_rs, cosi, t0):
    """ """
    inputs = [rp_over_rs, period, a_over_rs, cosi, t0]
    rp_over_rs, period, a_over_rs, cosi, t0 = [jnp.asarray(inp) for inp in inputs]
    Ttot = (
        period
        / jnp.pi
        * jnp.arcsin(
            jnp.sqrt((1.0 + rp_over_rs) ** 2 - (a_over_rs * cosi) ** 2)
            / jnp.sqrt(1.0 - cosi**2)
            / a_over_rs
        )
    )
    Tfull = (
        period
        / jnp.pi
        * jnp.arcsin(
            jnp.sqrt((1.0 - rp_over_rs) ** 2 - (a_over_rs * cosi) ** 2)
            / jnp.sqrt(1.0 - cosi**2)
            / a_over_rs
        )
    )
    t1 = t0 - Ttot / 2.0
    t2 = t0 - Tfull / 2.0
    t3 = t0 + Tfull / 2.0
    t4 = t0 + Ttot / 2.0
    return t1, t2, t3, t4


@jit
def calc_contact_times(rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0):
    """ """
    inputs = [rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0]
    rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0 = [
        jnp.asarray(inp) for inp in inputs
    ]
    f_init_t1 = (
        jnp.pi / 2.0
        - omega
        - jnp.arcsin(
            jnp.sqrt((1.0 + rp_over_rs) ** 2 - a_over_rs**2 * cosi**2)
            / a_over_rs
            / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_init_t2 = (
        jnp.pi / 2.0
        - omega
        - jnp.arcsin(
            jnp.sqrt((1.0 - rp_over_rs) ** 2 - a_over_rs**2 * cosi**2)
            / a_over_rs
            / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_init_t3 = (
        jnp.pi / 2.0
        - omega
        + jnp.arcsin(
            jnp.sqrt((1.0 - rp_over_rs) ** 2 - a_over_rs**2 * cosi**2)
            / a_over_rs
            / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_init_t4 = (
        jnp.pi / 2.0
        - omega
        + jnp.arcsin(
            jnp.sqrt((1.0 + rp_over_rs) ** 2 - a_over_rs**2 * cosi**2)
            / a_over_rs
            / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f1 = calc_true_anomaly_rsky(1 + rp_over_rs, f_init_t1, a_over_rs, ecc, omega, cosi)
    f2 = calc_true_anomaly_rsky(1 - rp_over_rs, f_init_t2, a_over_rs, ecc, omega, cosi)
    f3 = calc_true_anomaly_rsky(1 - rp_over_rs, f_init_t3, a_over_rs, ecc, omega, cosi)
    f4 = calc_true_anomaly_rsky(1 + rp_over_rs, f_init_t4, a_over_rs, ecc, omega, cosi)
    u1 = true_anomaly_to_eccentric_anomaly(f1, ecc)
    u2 = true_anomaly_to_eccentric_anomaly(f2, ecc)
    u3 = true_anomaly_to_eccentric_anomaly(f3, ecc)
    u4 = true_anomaly_to_eccentric_anomaly(f4, ecc)
    t1 = eccentric_anomaly_to_t_from_tperi(u1, ecc, period) + t0_to_tperi(
        t0, period, ecc, omega
    )
    t2 = eccentric_anomaly_to_t_from_tperi(u2, ecc, period) + t0_to_tperi(
        t0, period, ecc, omega
    )
    t3 = eccentric_anomaly_to_t_from_tperi(u3, ecc, period) + t0_to_tperi(
        t0, period, ecc, omega
    )
    t4 = eccentric_anomaly_to_t_from_tperi(u4, ecc, period) + t0_to_tperi(
        t0, period, ecc, omega
    )
    return t1, t2, t3, t4


@jit
def calc_true_anomaly_rsky(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi):
    """
    Calculate the true anomaly (f) from the rsky_over_rs.

    Parameters:
    rsky_over_rs (array): Projected distance between the planet and the host star.
    f_init (array) : Initial values for the true anomaly (f) in radians.
    a_over_rs (array): Semi-major axis normalized by the host star radius.
    ecc (array): Orbital eccentricity.
    omega (array): Argument of periastron in radians.
    cosi (array): Cosine of the orbital inclination.
    t0 (array): Time of inferior conjunction.

    Returns:
    array: True anomaly (f) in radians (vector output).
    """
    rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi = expand_arrays(
        rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi
    )

    def projected_distance_equation(f, rsky_over_rs, a_over_rs, ecc, omega, cosi):
        # Calculate the 3D distance based on the true anomaly
        r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f))
        # Compute the projected X and Y coordinates in the sky plane
        x = -r * jnp.cos(omega + f)
        y = -r * jnp.sin(omega + f) * cosi
        # Calculate the observed projected distance
        rsky = jnp.sqrt(x**2 + y**2)
        return rsky - rsky_over_rs

    # Vectorized root-finding
    def solve_for_f(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi):
        # Newton's method for root finding
        def newton_method(f):
            def func(f):
                return projected_distance_equation(
                    f, rsky_over_rs, a_over_rs, ecc, omega, cosi
                )

            def df(f):
                return jax.grad(func)(f)

            return f - func(f) / df(f)

        # Iterate with an initial guess
        f = f_init  # Initial estimate (choose the first value)
        for _ in range(20):  # Iterate until convergence
            f = newton_method(f)
        return f

    # Dynamically apply vmap based on input dimensions
    func = solve_for_f
    input_ndim = rsky_over_rs.ndim
    for _ in range(input_ndim):
        func = jax.vmap(func)
    # Process vector inputs
    f = func(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi)
    return f


@jit
def expand_arrays(*inputs):
    """ """
    arrays = [jnp.asarray(inp) for inp in inputs]
    # Determine the maximum number of dimensions among all inputs
    max_ndim = max(arr.ndim for arr in arrays)
    # If all inputs are scalars
    if max_ndim == 0:
        return arrays
    # Determine the maximum shape for each dimension
    max_shape = jnp.asarray(1)
    for arr in arrays:
        max_shape = max_shape * jnp.ones_like(arr)
    expanded_arrays = []
    for arr in arrays:
        expanded_arrays.append(arr * max_shape)
    return expanded_arrays
