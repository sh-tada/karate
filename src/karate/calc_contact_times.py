from karate.elements_to_orbit import true_anomaly_to_eccentric_anomaly
from karate.elements_to_orbit import t0_to_tperi
from karate.elements_to_orbit import eccentric_anomaly_to_t_from_tperi
import jax.numpy as jnp
import jax
from jax import jit


@jit
def calc_contact_times_circular(rp_over_rs, period, a_over_rs, cosi, t0):
    """
    Calculate the contact times for a circular orbit, based on the planet's
    radius and orbital parameters.

    Parameters
    ----------
    rp_over_rs : float or array-like
        Planetary radius normalized by the stellar radius.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.

    Returns
    -------
    tuple
        A tuple containing the contact times: the beginning and end of the
        ingress and egress phases of the planetary transit.
    """
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
    """
    Calculate the contact times for an eccentric orbit, based on the planet's
    radius and orbital parameters.

    Parameters
    ----------
    rp_over_rs : float or array-like
        Planetary radius normalized by the stellar radius.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.

    Returns
    -------
    tuple
        A tuple containing the contact times: the beginning and end of the
        ingress and egress phases of the planetary transit.
    """
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
def calc_contact_times_with_deltac(
    rp_over_rs, dc_over_rs_x, dc_over_rs_y, period, a_over_rs, ecc, omega, cosi, t0
):
    """
    Calculate the contact times for an eccentric orbit with a displacement of
    the planetary shadow, based on the planet's radius and orbital parameters.

    This function computes the contact times (the beginning and end of the
    ingress and egress phases) of the planetary transit, considering the
    displacement of the planetary shadow in the sky plane.

    Parameters
    ----------
    rp_over_rs : float or array-like
        Planetary radius normalized by the stellar radius.
    dc_over_rs_x : float or array-like
        Displacement in the x-direction normalized by the stellar radius.
    dc_over_rs_y : float or array-like
        Displacement in the y-direction normalized by the stellar radius.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination angle.
    t0 : float or array-like
        Time of inferior conjunction.

    Returns
    -------
    tuple
        A tuple containing the contact times: the beginning and end of the
        ingress and egress phases of the planetary transit, considering the
        displacement of the planetary shadow.
    """
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
    f1 = calc_true_anomaly_rsky_with_deltac(
        1 + rp_over_rs,
        dc_over_rs_x,
        dc_over_rs_y,
        f_init_t1,
        a_over_rs,
        ecc,
        omega,
        cosi,
    )
    f2 = calc_true_anomaly_rsky_with_deltac(
        1 - rp_over_rs,
        dc_over_rs_x,
        dc_over_rs_y,
        f_init_t2,
        a_over_rs,
        ecc,
        omega,
        cosi,
    )
    f3 = calc_true_anomaly_rsky_with_deltac(
        1 - rp_over_rs,
        dc_over_rs_x,
        dc_over_rs_y,
        f_init_t3,
        a_over_rs,
        ecc,
        omega,
        cosi,
    )
    f4 = calc_true_anomaly_rsky_with_deltac(
        1 + rp_over_rs,
        dc_over_rs_x,
        dc_over_rs_y,
        f_init_t4,
        a_over_rs,
        ecc,
        omega,
        cosi,
    )
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


def projected_distance_equation(f, rsky_over_rs, a_over_rs, ecc, omega, cosi):
    # Calculate the 3D distance based on the true anomaly
    r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f))
    # Compute the projected X and Y coordinates in the sky plane
    x = -r * jnp.cos(omega + f)
    y = -r * jnp.sin(omega + f) * cosi
    # Calculate the observed projected distance
    rsky_tmp = jnp.sqrt(x**2 + y**2)
    return rsky_tmp - rsky_over_rs


# Vectorized root-finding
def solve_for_f(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi):
    # Newton's method for root finding
    func = projected_distance_equation
    df = jax.grad(projected_distance_equation)

    # Newton's method update step
    def newton_step(f, _):
        f_next = f - func(f, rsky_over_rs, a_over_rs, ecc, omega, cosi) / df(
            f, rsky_over_rs, a_over_rs, ecc, omega, cosi
        )
        return f_next, None

    # Use lax.scan for iterative updates
    f_final, _ = jax.lax.scan(newton_step, f_init, None, length=20)
    return f_final


def calc_true_anomaly_rsky(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi):
    """
    Calculate the true anomaly from the projected distance in the sky plane.

    Parameters
    ----------
    rsky_over_rs : array
        Projected distance between the planet and the host star, normalized by
        the stellar radius.
    f_init : array
        Initial guess for the true anomaly in radians.
    a_over_rs : array
        Semi-major axis normalized by the stellar radius.
    ecc : array
        Orbital eccentricity.
    omega : array
        Argument of periastron in radians.
    cosi : array
        Cosine of the orbital inclination.

    Returns
    -------
    array
        The true anomaly in radians that corresponds to the given rsky_over_rs.
    """
    rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi = expand_arrays(
        rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi
    )

    # Dynamically apply vmap based on input dimensions
    func = solve_for_f
    input_ndim = rsky_over_rs.ndim
    for _ in range(input_ndim):
        func = jax.vmap(func)
    # Process vector inputs
    f = func(rsky_over_rs, f_init, a_over_rs, ecc, omega, cosi)
    return f


def projected_distance_equation_deltac(
    f, rsky_over_rs, dc_over_rs_x, dc_over_rs_y, a_over_rs, ecc, omega, cosi
):
    # Calculate the 3D distance based on the true anomaly
    r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f))
    # Compute the projected X and Y coordinates in the sky plane
    x = -r * jnp.cos(omega + f) + dc_over_rs_x
    y = -r * jnp.sin(omega + f) * cosi + dc_over_rs_y
    # Calculate the observed projected distance
    rsky_tmp = jnp.sqrt(x**2 + y**2)
    return rsky_tmp - rsky_over_rs


# Vectorized root-finding
def solve_for_f_deltac(
    rsky_over_rs, dc_over_rs_x, dc_over_rs_y, f_init, a_over_rs, ecc, omega, cosi
):
    # Newton's method for root finding
    func = projected_distance_equation_deltac
    df = jax.grad(projected_distance_equation_deltac)

    # Newton's method update step
    def newton_step(f, _):
        f_next = f - func(
            f, rsky_over_rs, dc_over_rs_x, dc_over_rs_y, a_over_rs, ecc, omega, cosi
        ) / df(f, rsky_over_rs, dc_over_rs_x, dc_over_rs_y, a_over_rs, ecc, omega, cosi)
        return f_next, None

    # Use lax.scan for iterative updates
    f_final, _ = jax.lax.scan(newton_step, f_init, None, length=20)

    return f_final


def calc_true_anomaly_rsky_with_deltac(
    rsky_over_rs, dc_over_rs_x, dc_over_rs_y, f_init, a_over_rs, ecc, omega, cosi
):
    """
    Calculate the true anomaly from the projected distance in the sky plane,
    accounting for the displacement of the planetary shadow.

    Parameters
    ----------
    rsky_over_rs : array
        Projected distance between the planet and the host star, normalized by
        the stellar radius.
    dc_over_rs_x : float or array-like
        Displacement in the x-direction normalized by the stellar radius.
    dc_over_rs_y : float or array-like
        Displacement in the y-direction normalized by the stellar radius.
    f_init : array
        Initial guess for the true anomaly in radians.
    a_over_rs : array
        Semi-major axis normalized by the stellar radius.
    ecc : array
        Orbital eccentricity.
    omega : array
        Argument of periastron in radians.
    cosi : array
        Cosine of the orbital inclination.

    Returns
    -------
    array
        The true anomaly in radians that corresponds to the given rsky_over_rs,
        considering the displacement of the planetary shadow.
    """
    rsky_over_rs, dc_over_rs_x, dc_over_rs_y, f_init, a_over_rs, ecc, omega, cosi = (
        expand_arrays(
            rsky_over_rs,
            dc_over_rs_x,
            dc_over_rs_y,
            f_init,
            a_over_rs,
            ecc,
            omega,
            cosi,
        )
    )

    # Dynamically apply vmap based on input dimensions
    func = solve_for_f_deltac
    input_ndim = rsky_over_rs.ndim
    for _ in range(input_ndim):
        func = jax.vmap(func)
    # Process vector inputs
    f = func(
        rsky_over_rs, dc_over_rs_x, dc_over_rs_y, f_init, a_over_rs, ecc, omega, cosi
    )
    return f


def expand_arrays(*inputs):
    """
    Expand the input arrays to have the same shape, broadcasting them as needed.

    Parameters
    ----------
    *inputs : Variable number of input arrays to be expanded.

    Returns
    -------
    tuple
        A tuple of expanded arrays with matching shapes.
    """
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
