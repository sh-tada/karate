import numpy as np
import jax.numpy as jnp
from jax import random

# PPL
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist


def coordinates_circular_orbit(t, period, a_over_rs, cosi, t0):
    """
    Compute the x and y coordinates of a planet in a circular orbit.

    This function calculates the position of a planet in a circular
    orbit at a given time `t`. The orbit is described by the orbital
    period, semi-major axis, and inclination.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to calculate the planet's coordinates.
    period : float
        Orbital period of the planet.
    a_over_rs : float
        Semi-major axis divided by stellar radius.
    cosi : float
        Cosine of the orbital inclination angle.
    t0 : float
        Time of conjunction (when the planet passes closest to the star's center).

    Returns
    -------
    x : float or array-like
        X-coordinate(s) of the planet's position (in units of stellar radii).
    y : float or array-like
        Y-coordinate(s) of the planet's position (in units of stellar radii).
    """
    x = a_over_rs * jnp.sin(2 * jnp.pi * (t - t0) / period)
    y = -a_over_rs * jnp.cos(2 * jnp.pi * (t - t0) / period) * cosi
    return x, y


def params_from_rp_small(
    wavelength, rp_over_rs_samples, a_over_rs_samples, cosi_samples, t0_samples
):
    """
    Extract orbital parameters based on small planetary radius samples.

    This function identifies the samples corresponding to small
    planetary radii (in the bottom 10% of the distribution) and
    computes the average values of the orbital parameters (semi-major
    axis, orbital inclination, and mid-transit time) for these samples.

    Parameters
    ----------
    wavelength : array-like
        Array of wavelengths.
    rp_over_rs_samples : array-like
        Samples of planetary radius divided by stellar radius.
    a_over_rs_samples : array-like
        Samples of semi-major axis divided by stellar radius.
    cosi_samples : array-like
        Samples of cosine of the orbital inclination angle.
    t0_samples : array-like
        Samples of the mid-transit time.

    Returns
    -------
    a_over_rs_b : float
        Average semi-major axis to stellar radius ratio for the small-radius samples.
    cosi_b : float
        Average cosine of the orbital inclination for the small-radius samples.
    t0_b : float
        Average mid-transit time for the small-radius samples.
    lambda0 : float
        Wavelength corresponding to the small-radius samples.
    """
    rp_median = np.median(rp_over_rs_samples, axis=0)
    small_radii_indices = np.where(rp_median <= np.percentile(rp_median, 10))

    a_over_rs_b = np.mean(np.median(a_over_rs_samples, axis=0)[small_radii_indices])
    cosi_b = np.mean(np.median(cosi_samples, axis=0)[small_radii_indices])
    t0_b = np.mean(np.median(t0_samples, axis=0)[small_radii_indices])
    lambda0 = np.mean(wavelength[small_radii_indices])
    return a_over_rs_b, cosi_b, t0_b, lambda0


def ti_te_tau_symmetrical(
    rp_over_rs,
    period,
    a_over_rs_b,
    cosi_b,
    t0_b,
    rs_alpha=1,
):
    """
    Calculate ingress and egress times and transit duration for
    completely symmetrical atmosphere.

    This function computes the ingress time (`ti`), egress time (`te`),
    and the duration of ingress/egress (`tau`) based on the planetary
    radius, orbital period, and orbital geometry. The results are derived
    under the assumption of no chromatic variations of Ttot, Tfull, and t0.

    Parameters
    ----------
    rp_over_rs : float or array-like
        Ratio of planetary radius to stellar radius.
    period : float
        Orbital period of the planet.
    a_over_rs_b : float
        Semi-major axis divided by stellar radius for the center of mass.
    cosi_b : float
        Cosine of the orbital inclination angle for the center of mass.
    t0_b : float
        Mid-transit time of the planet for the center of mass.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at a given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    ti_from_rp : float or array-like
        Ingress time of the planet based on its radius.
    te_from_rp : float or array-like
        Egress time of the planet based on its radius.
    tau_from_rp : float or array-like
        duration of ingress or egress based on its radius.
    """
    rs_alpha = np.asarray(rs_alpha)
    if rs_alpha.ndim != 0:
        rs_alpha = rs_alpha[None, :]

    Ttot_from_rp = (
        period
        / np.pi
        * np.arcsin(
            np.sqrt((rs_alpha**2 * (1 + rp_over_rs) ** 2 - (a_over_rs_b * cosi_b) ** 2))
            / np.sqrt(1 - cosi_b**2)
            / a_over_rs_b
        )
    )
    Tfull_from_rp = (
        period
        / np.pi
        * np.arcsin(
            np.sqrt((rs_alpha**2 * (1 - rp_over_rs) ** 2 - (a_over_rs_b * cosi_b) ** 2))
            / np.sqrt(1 - cosi_b**2)
            / a_over_rs_b
        )
    )
    ti_from_rp = ((t0_b - Ttot_from_rp / 2) + (t0_b - Tfull_from_rp / 2)) / 2
    te_from_rp = ((t0_b + Ttot_from_rp / 2) + (t0_b + Tfull_from_rp / 2)) / 2
    tau_from_rp = (Ttot_from_rp - Tfull_from_rp) / 2
    return ti_from_rp, te_from_rp, tau_from_rp


if __name__ == "__main__":
    pass
