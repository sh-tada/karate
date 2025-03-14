from karate.elements_to_orbit import orbital_elements_to_coordinates
from karate.elements_to_orbit import orbital_elements_to_coordinates_circular
from karate.calc_contact_times import calc_true_anomaly_rsky
import jax.numpy as jnp
from jax import jit


# @jit
def cb_to_delta_c_ingress(cb_X_t1, cb_Y_t1, cb_X_t2, cb_Y_t2, k, rs_alpha=1):
    """
    Calculate the center shift of the planetary shadow at ingress.

    This function computes the displacement of the planetary shadow's center
    during the ingress phase, based on the center of mass coordinates
    at two different contact times (t1 and t2). The displacement depends on the
    ratio of planetary to stellar radius and the relative stellar radius
    at a given wavelength.

    Parameters
    ----------
    cb_X_t1 : float or array-like
        X-coordinate of the center of mass at contact time t1
        (in stellar radii at the reference wavelength).
    cb_Y_t1 : float or array-like
        Y-coordinate of the center of mass at contact time t1
        (in stellar radii at the reference wavelength).
    cb_X_t2 : float or array-like
        X-coordinate of the center of mass at contact time t2
        (in stellar radii at the reference wavelength).
    cb_Y_t2 : float or array-like
        Y-coordinate of the center of mass at contact time t2
        (in stellar radii at the reference wavelength).
    k : float or array-like
        Ratio of planetary radius to stellar radius at a given wavelength.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X : float or array-like
        X-coordinate of the center displacement
        (in units of stellar radii at the reference wavelength).
    dc_Y : float or array-like
        Y-coordinate of the center displacement
        (in units of stellar radii at the reference wavelength).
    """

    m_X = (cb_X_t1 + cb_X_t2) / 2.0
    m_Y = (cb_Y_t1 + cb_Y_t2) / 2.0
    d_X = (cb_X_t1 - cb_X_t2) / 2.0
    d_Y = (cb_Y_t1 - cb_Y_t2) / 2.0
    d = jnp.sqrt(d_X**2 + d_Y**2)
    d_X_n = d_X / d
    d_Y_n = d_Y / d
    # dcx and dcy are NaN when d < rs_alpha * k
    # Replace NaN with the values for the case d = rs_alpha * k
    print(
        "The number of d < rs_alpha*k (ingress)",
        jnp.count_nonzero(jnp.atleast_1d(d - rs_alpha * k) < 0, axis=0),
    )
    # print("The number of NaN (ingress)", jnp.count_nonzero(d - rs_alpha * k < 0))
    # print("The index of NaN (ingress)", jnp.argwhere(d - rs_alpha * k < 0))
    s_X = rs_alpha**2 * k / d
    s_Y = (rs_alpha**2 - d**2) * (1.0 - rs_alpha**2 * (k / d) ** 2)
    s_Y = jnp.sqrt(jnp.where(s_Y >= 0, s_Y, 0))

    dc_X_ingress = -m_X + d_X_n * s_X - d_Y_n * s_Y
    dc_Y_ingress = -m_Y + d_Y_n * s_X + d_X_n * s_Y
    # dc_X_ingress = jnp.nan_to_num(dc_X_ingress, nan=-m_X + d_X_n * rs_alpha)
    # dc_Y_ingress = jnp.nan_to_num(dc_Y_ingress, nan=-m_Y + d_Y_n * rs_alpha)
    return dc_X_ingress, dc_Y_ingress


# @jit
def cb_to_delta_c_egress(cb_X_t3, cb_Y_t3, cb_X_t4, cb_Y_t4, k, rs_alpha=1):
    """
    Calculate the center displacement of the planetary shadow at egress.

    This function computes the displacement of the planetary shadow's center
    during the egress phase, based on the center of mass coordinates at
    two different contact times (t3 and t4). The displacement depends on the
    ratio of planetary to stellar radius and the relative stellar radius
    at a given wavelength.

    Parameters
    ----------
    cb_X_t3 : float or array-like
        X-coordinate of the center of mass at contact time t3
        (in stellar radii at the reference wavelength).
    cb_Y_t3 : float or array-like
        Y-coordinate of the center of mass at contact time t3
        (in stellar radii at the reference wavelength).
    cb_X_t4 : float or array-like
        X-coordinate of the center of mass at contact time t4
        (in stellar radii at the reference wavelength).
    cb_Y_t4 : float or array-like
        Y-coordinate of the center of mass at contact time t4
        (in stellar radii at the reference wavelength).
    k : float or array-like
        Ratio of planetary radius to stellar radius at a given wavelength.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X : float or array-like
        X-coordinate of the center displacement
        (in units of stellar radii at the reference wavelength).
    dc_Y : float or array-like
        Y-coordinate of the center displacement
        (in units of stellar radii at the reference wavelength).
    """
    m_X = (cb_X_t4 + cb_X_t3) / 2
    m_Y = (cb_Y_t4 + cb_Y_t3) / 2
    d_X = (cb_X_t4 - cb_X_t3) / 2
    d_Y = (cb_Y_t4 - cb_Y_t3) / 2
    d = jnp.sqrt(d_X**2 + d_Y**2)
    d_X_n = d_X / d
    d_Y_n = d_Y / d
    # dcx and dcy are NaN when d < rs_alpha * k
    # Replace NaN with the values for the case d = rs_alpha * k
    print(
        "The number of d < rs_alpha*k (egress)",
        jnp.count_nonzero(jnp.atleast_1d(d - rs_alpha * k) < 0, axis=0),
    )
    # print("The number of NaN (egress)", jnp.count_nonzero(d - rs_alpha * k < 0))
    # print("The index of NaN (egress)", jnp.argwhere(d - rs_alpha * k < 0))
    s_X = rs_alpha**2 * k / d
    s_Y = (rs_alpha**2 - d**2) * (1.0 - rs_alpha**2 * (k / d) ** 2)
    s_Y = -jnp.sqrt(jnp.where(s_Y >= 0, s_Y, 0))

    dc_X_egress = -m_X + d_X_n * s_X - d_Y_n * s_Y
    dc_Y_egress = -m_Y + d_Y_n * s_X + d_X_n * s_Y
    # dc_X_egress = jnp.nan_to_num(dc_X_egress, nan=-m_X + d_X_n * rs_alpha)
    # dc_Y_egress = jnp.nan_to_num(dc_Y_egress, nan=-m_Y + d_Y_n * rs_alpha)
    return dc_X_egress, dc_Y_egress


# @jit
def contact_times_to_delta_c_ingress(
    k_lambda, t1_lambda, t2_lambda, period, a_over_rs, ecc, omega, cosi, t0, rs_alpha=1
):
    """
    Calculate the center displacement at ingress by computing the coordinates
    at two different contact times.

    Parameters
    ----------
    k_lambda : float or array-like
        Ratio of planetary radius to stellar radius at the given wavelength.
    t1_lambda : float or array-like
        Contact time t1 at the given wavelength.
    t2_lambda : float or array-like
        Contact time t2 at the given wavelength.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X_ingress : float or array-like
        X-coordinate of the center displacement at ingress.
    dc_Y_ingress : float or array-like
        Y-coordinate of the center displacement at ingress.
    """
    cb_X_t1, cb_Y_t1 = orbital_elements_to_coordinates(
        t1_lambda, period, a_over_rs, ecc, omega, cosi, t0
    )
    cb_X_t2, cb_Y_t2 = orbital_elements_to_coordinates(
        t2_lambda, period, a_over_rs, ecc, omega, cosi, t0
    )
    dc_X_ingress, dc_Y_ingress = cb_to_delta_c_ingress(
        cb_X_t1, cb_Y_t1, cb_X_t2, cb_Y_t2, k_lambda, rs_alpha
    )
    return dc_X_ingress, dc_Y_ingress


# @jit
def contact_times_to_delta_c_egress(
    k_lambda, t3_lambda, t4_lambda, period, a_over_rs, ecc, omega, cosi, t0, rs_alpha=1
):
    """
    Calculate the center displacement at egress by computing the coordinates
    at two different contact times.

    Parameters
    ----------
    k_lambda : float or array-like
        Ratio of planetary radius to stellar radius at the given wavelength.
    t3_lambda : float or array-like
        Contact time t3 at the given wavelength.
    t4_lambda : float or array-like
        Contact time t4 at the given wavelength.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X_egress : float or array-like
        X-coordinate of the center displacement at egress.
    dc_Y_egress : float or array-like
        Y-coordinate of the center displacement at egress.
    """
    cb_X_t3, cb_Y_t3 = orbital_elements_to_coordinates(
        t3_lambda, period, a_over_rs, ecc, omega, cosi, t0
    )
    cb_X_t4, cb_Y_t4 = orbital_elements_to_coordinates(
        t4_lambda, period, a_over_rs, ecc, omega, cosi, t0
    )
    dc_X_egress, dc_Y_egress = cb_to_delta_c_egress(
        cb_X_t3, cb_Y_t3, cb_X_t4, cb_Y_t4, k_lambda, rs_alpha
    )
    return dc_X_egress, dc_Y_egress


@jit
def rotate_delta_c_ingress(dc_X_ingress, dc_Y_ingress, a_over_rs, ecc, omega, cosi):
    """
    Rotate the center displacement at ingress to obtain xi- and yi-coordinates.

    This function applies the rotation of the center displacement for the
    ingress phase to obtain xi- and yi-coordinates.

    Parameters
    ----------
    dc_X_ingress : float or array-like
        X-coordinate of the center displacement at ingress.
    dc_Y_ingress : float or array-like
        Y-coordinate of the center displacement at ingress.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination.

    Returns
    -------
    dc_xi_ingress : float or array-like
        xi-coordinate of the center displacement at ingress.
    dc_yi_ingress : float or array-like
        yi-coordinate of the center displacement at ingress.
    """
    f_init = (
        jnp.pi / 2.0
        - omega
        - jnp.arcsin(
            jnp.sqrt(1.0 - a_over_rs**2 * cosi**2) / a_over_rs / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_ib = calc_true_anomaly_rsky(1.0, f_init, a_over_rs, ecc, omega, cosi)
    r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f_ib))
    cb_X_tib = -r * jnp.cos(omega + f_ib)
    cb_Y_tib = -r * jnp.sin(omega + f_ib) * cosi
    dc_xi_ingress = -(dc_X_ingress * cb_X_tib + dc_Y_ingress * cb_Y_tib)
    dc_yi_ingress = -(-dc_X_ingress * cb_Y_tib + dc_Y_ingress * cb_X_tib)
    return dc_xi_ingress, dc_yi_ingress


@jit
def rotate_delta_c_egress(dc_X_egress, dc_Y_egress, a_over_rs, ecc, omega, cosi):
    """
    Rotate the center displacement at egress to obtain xe- and ye-coordinates.

    This function applies the rotation of the center displacement for the
    egress phase to obtain xe- and ye-coordinates.

    Parameters
    ----------
    dc_X_egress : float or array-like
        X-coordinate of the center displacement at egress.
    dc_Y_egress : float or array-like
        Y-coordinate of the center displacement at egress.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination.

    Returns
    -------
    dc_xe_egress : float or array-like
        xe-coordinate of the center displacement at egress.
    dc_ye_egress : float or array-like
        ye-coordinate of the center displacement at egress.
    """
    f_init = (
        jnp.pi / 2.0
        - omega
        + jnp.arcsin(
            jnp.sqrt(1.0 - a_over_rs**2 * cosi**2) / a_over_rs / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_eb = calc_true_anomaly_rsky(1.0, f_init, a_over_rs, ecc, omega, cosi)
    r = a_over_rs * (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f_eb))
    cb_X_teb = -r * jnp.cos(omega + f_eb)
    cb_Y_teb = -r * jnp.sin(omega + f_eb) * cosi
    dc_xe_egress = dc_X_egress * cb_X_teb + dc_Y_egress * cb_Y_teb
    dc_ye_egress = -dc_X_egress * cb_Y_teb + dc_Y_egress * cb_X_teb
    return dc_xe_egress, dc_ye_egress


# @jit
def contact_times_to_delta_c_ingress_circular(
    k_lambda, t1_lambda, t2_lambda, period, a_over_rs, cosi, t0, rs_alpha=1
):
    """
    Calculate the center displacement at ingress for a circular orbit by
    computing the coordinates at two different contact times.

    Parameters
    ----------
    k_lambda : float or array-like
        Ratio of planetary radius to stellar radius at the given wavelength.
    t1_lambda : float or array-like
        Contact time t1 at the given wavelength.
    t2_lambda : float or array-like
        Contact time t2 at the given wavelength.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X_ingress : float or array-like
        X-coordinate of the center displacement at ingress.
    dc_Y_ingress : float or array-like
        Y-coordinate of the center displacement at ingress.
    """
    cb_X_t1, cb_Y_t1 = orbital_elements_to_coordinates_circular(
        t1_lambda, period, a_over_rs, cosi, t0
    )
    cb_X_t2, cb_Y_t2 = orbital_elements_to_coordinates_circular(
        t2_lambda, period, a_over_rs, cosi, t0
    )
    dc_X_ingress, dc_Y_ingress = cb_to_delta_c_ingress(
        cb_X_t1, cb_Y_t1, cb_X_t2, cb_Y_t2, k_lambda, rs_alpha
    )
    return dc_X_ingress, dc_Y_ingress


# @jit
def contact_times_to_delta_c_egress_circular(
    k_lambda, t3_lambda, t4_lambda, period, a_over_rs, cosi, t0, rs_alpha=1
):
    """
    Calculate the center displacement at egress for a circular orbit by
    computing the coordinates at two different contact times.

    Parameters
    ----------
    k_lambda : float or array-like
        Ratio of planetary radius to stellar radius at the given wavelength.
    t3_lambda : float or array-like
        Contact time t3 at the given wavelength.
    t4_lambda : float or array-like
        Contact time t4 at the given wavelength.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination.
    t0 : float or array-like
        Time of inferior conjunction.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dc_X_egress : float or array-like
        X-coordinate of the center displacement at egress.
    dc_Y_egress : float or array-like
        Y-coordinate of the center displacement at egress.
    """
    cb_X_t3, cb_Y_t3 = orbital_elements_to_coordinates_circular(
        t3_lambda, period, a_over_rs, cosi, t0
    )
    cb_X_t4, cb_Y_t4 = orbital_elements_to_coordinates_circular(
        t4_lambda, period, a_over_rs, cosi, t0
    )
    dc_X_egress, dc_Y_egress = cb_to_delta_c_egress(
        cb_X_t3, cb_Y_t3, cb_X_t4, cb_Y_t4, k_lambda, rs_alpha
    )
    return dc_X_egress, dc_Y_egress


@jit
def rotate_delta_c_ingress_circular(dc_X_ingress, dc_Y_ingress, a_over_rs, cosi):
    """
    Rotate the center displacement at ingress for a circular orbit based on
    orbital parameters to obtain xi- and yi-coordinates..

    This function applies the rotation of the center displacement for the
    ingress phase to obtain xi- and yi-coordinates.

    Parameters
    ----------
    dc_X_ingress : float or array-like
        X-coordinate of the center displacement at ingress.
    dc_Y_ingress : float or array-like
        Y-coordinate of the center displacement at ingress.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination.

    Returns
    -------
    dc_xi_ingress : float or array-like
        xi-coordinate of the center displacement at ingress.
    dc_yi_ingress : float or array-like
        yi-coordinate of the center displacement at ingress.
    """
    f_ib = jnp.pi / 2.0 - jnp.arcsin(
        jnp.sqrt((1.0 - (a_over_rs * cosi) ** 2)) / jnp.sqrt(1.0 - cosi**2) / a_over_rs
    )
    cb_X_tib = -a_over_rs * jnp.cos(f_ib)
    cb_Y_tib = -a_over_rs * jnp.sin(f_ib) * cosi
    dc_xi_ingress = -(dc_X_ingress * cb_X_tib + dc_Y_ingress * cb_Y_tib)
    dc_yi_ingress = -(-dc_X_ingress * cb_Y_tib + dc_Y_ingress * cb_X_tib)
    return dc_xi_ingress, dc_yi_ingress


@jit
def rotate_delta_c_egress_circular(dc_X_egress, dc_Y_egress, a_over_rs, cosi):
    """
    Rotate the center displacement at egress for a circular orbit based on
    orbital parameters to obtain xe- and ye-coordinates.

    This function applies the rotation of the center displacement for the
    egress phase to obtain xe- and ye-coordinates.

    Parameters
    ----------
    dc_X_egress : float or array-like
        X-coordinate of the center displacement at egress.
    dc_Y_egress : float or array-like
        Y-coordinate of the center displacement at egress.
    a_over_rs : float or array-like
        Semi-major axis normalized by stellar radius.
    cosi : float or array-like
        Cosine of the orbital inclination.

    Returns
    -------
    dc_xe_egress : float or array-like
        xe-coordinate of the center displacement at egress.
    dc_ye_egress : float or array-like
        ye-coordinate of the center displacement at egress.
    """
    f_eb = jnp.pi / 2 + jnp.arcsin(
        jnp.sqrt((1.0 - (a_over_rs * cosi) ** 2)) / jnp.sqrt(1.0 - cosi**2) / a_over_rs
    )
    cb_X_teb = -a_over_rs * jnp.cos(f_eb)
    cb_Y_teb = -a_over_rs * jnp.sin(f_eb) * cosi
    dc_xe_egress = dc_X_egress * cb_X_teb + dc_Y_egress * cb_Y_teb
    dc_ye_egress = -dc_X_egress * cb_Y_teb + dc_Y_egress * cb_X_teb
    return dc_xe_egress, dc_ye_egress


@jit
def dcx_to_rp_spectra(k, dc_x, rs_alpha=1):
    """
    Convert the center displacement in the x-direction to the projected
    planetary radii in the positive and negative x directions at a given
    wavelength.

    This function calculates the projected planetary radius in the positive
    (xp) and negative (xn) x directions at a given wavelength, based on the
    center displacement in the x-direction and the ratio of planetary radius
    to stellar radius.

    Parameters
    ----------
    k : float or array-like
        Ratio of planetary radius to stellar radius at the given wavelength.
    dc_x : float or array-like
        Center displacement in the x-direction (in units of stellar radii).
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    rp_xp : float or array-like
        Projected planetary radius in the positive x-direction (xp) at the
        given wavelength.
    rp_xn : float or array-like
        Projected planetary radius in the negative x-direction (xn) at the
        given wavelength.
    """
    rp_xp = rs_alpha * k + dc_x
    rp_xn = rs_alpha * k - dc_x
    return rp_xp, rp_xn
