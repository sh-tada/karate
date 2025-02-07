from karate.elements_to_orbit import true_anomaly_to_eccentric_anomaly
from karate.calc_contact_times import calc_true_anomaly_rsky
import jax.numpy as jnp
from jax import jit


@jit
def duration_Tc_from_rs_alpha(
    rs_alpha,
    period,
    a_over_rs,
    ecc,
    omega,
    cosi,
):
    """
    Calculate the duration of the planetary transit (Tc) based on the stellar
    radius at a given wavelength.

    This function computes the duration of the planetary transit as a function
    of the stellar radius (rs_alpha) and orbital parameters, including
    eccentricity and inclination.

    Parameters
    ----------
    rs_alpha : float or array-like
        Stellar radius at the given wavelength, normalized by that at the
        reference wavelength.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius at the reference
        wavelength.
    ecc : float or array-like
        Orbital eccentricity.
    omega : float or array-like
        Argument of periastron in radians.
    cosi : float or array-like
        Cosine of the orbital inclination angle.

    Returns
    -------
    duration : float or array-like
        Duration of the planetary transit (Tc), expressed in the same units
        as the orbital period.
    """
    f_init = (
        jnp.pi / 2.0
        - omega
        - jnp.arcsin(
            jnp.sqrt(1.0 - a_over_rs**2 * cosi**2) / a_over_rs / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_ib = calc_true_anomaly_rsky(rs_alpha, f_init, a_over_rs, ecc, omega, cosi)
    u_ib = true_anomaly_to_eccentric_anomaly(f_ib, ecc)

    f_init = (
        jnp.pi / 2.0
        - omega
        + jnp.arcsin(
            jnp.sqrt(1.0 - a_over_rs**2 * cosi**2) / a_over_rs / jnp.sqrt(1.0 - cosi**2)
        )
    )
    f_eb = calc_true_anomaly_rsky(rs_alpha, f_init, a_over_rs, ecc, omega, cosi)
    u_eb = true_anomaly_to_eccentric_anomaly(f_eb, ecc)

    duration = ((u_eb - u_ib) % (2.0 * jnp.pi)) / jnp.pi / 2.0 * period
    return duration


def rs_alpha_from_duration_Tc_circular(
    Tc,
    period,
    a_over_rs,
    cosi,
):
    """
    Calculate the stellar radius at a given wavelength relative to the
    reference wavelength based on the duration of the planetary transit
    for a circular orbit.

    This function computes the stellar radius (rs_alpha) at the given
    wavelength based on the transit duration (Tc) for a circular orbit.
    The relationship between transit duration and stellar radius is used.

    Parameters
    ----------
    Tc : float or array-like
        Duration of the planetary transit.
    period : float or array-like
        Orbital period of the planet.
    a_over_rs : float or array-like
        Semi-major axis normalized by the stellar radius at the reference
        wavelength.
    cosi : float or array-like
        Cosine of the orbital inclination angle.

    Returns
    -------
    rs_alpha : float or array-like
        Stellar radius at the given wavelength, normalized by that at the
        reference wavelength.
    """
    # Conversion from the duration to Rs
    theta_lambda = jnp.pi * Tc / period
    rs_alpha = a_over_rs * jnp.sqrt(
        jnp.sin(theta_lambda) ** 2 + jnp.cos(theta_lambda) ** 2 * cosi**2
    )
    return rs_alpha
