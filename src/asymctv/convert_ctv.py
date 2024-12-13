import numpy as np
import jax.numpy as jnp
from jax import random

# PPL
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist


def delta_c_ingress(cbx_t1, cby_t1, cbx_t2, cby_t2, k, rs_alpha=1):
    """
    Calculate the center shift of the planetary shadow at ingress.

    This function computes the shift of the planetary shadow's center
    during the ingress phase, based on the center of mass coordinates
    at two different contact times (t1 and t2). The shift depends on the
    ratio of planetary to stellar radius and the relative stellar radius
    at a given wavelength.

    Parameters
    ----------
    cbx_t1 : float or array-like
        X-coordinate of the center of mass at contact time t1
        (in stellar radii at the reference wavelength).
    cby_t1 : float or array-like
        Y-coordinate of the center of mass at contact time t1
        (in stellar radii at the reference wavelength).
    cbx_t2 : float or array-like
        X-coordinate of the center of mass at contact time t2
        (in stellar radii at the reference wavelength).
    cby_t2 : float or array-like
        Y-coordinate of the center of mass at contact time t2
        (in stellar radii at the reference wavelength).
    k : float or array-like
        Ratio of planetary radius to stellar radius at a given wavelength.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dcx : float or array-like
        X-coordinate of the center shift
        (in units of stellar radii at the reference wavelength).
    dcy : float or array-like
        Y-coordinate of the center shift
        (in units of stellar radii at the reference wavelength).
    """

    rs_alpha = np.asarray(rs_alpha)
    if rs_alpha.ndim != 0:
        rs_alpha = rs_alpha[None, :]
    mx = (cbx_t1 + cbx_t2) / 2
    my = (cby_t1 + cby_t2) / 2
    dx = (cbx_t1 - cbx_t2) / 2
    dy = (cby_t1 - cby_t2) / 2
    d = np.sqrt(dx**2 + dy**2)
    dx_n = dx / d
    dy_n = dy / d
    dcx = (
        -mx
        + dx_n * rs_alpha**2 * k / d
        - dy_n * jnp.sqrt((rs_alpha**2 - d**2) * (1 - rs_alpha**2 * (k / d) ** 2))
    )
    dcy = (
        -my
        + dy_n * rs_alpha**2 * k / d
        + dx_n * jnp.sqrt((rs_alpha**2 - d**2) * (1 - rs_alpha**2 * (k / d) ** 2))
    )
    # dcx and dcy are NaN when (1 - rs_alpha**2 * (k / d) ** 2) < 0
    # Replace NaN with min(dcx) and max(dcy)
    print("The number of NaN (ingress)", len(np.where(np.isnan(dcx))[0]))
    print("The index of NaN (ingress)", np.where(np.isnan(dcx)))
    dcx = np.nan_to_num(dcx, nan=np.nanmin(dcx))
    dcy = np.nan_to_num(dcy, nan=np.nanmax(dcy))
    return dcx, dcy


def delta_c_egress(cbx_t3, cby_t3, cbx_t4, cby_t4, k, rs_alpha=1):
    """
    Calculate the center shift of the planetary shadow at egress.

    This function computes the shift of the planetary shadow's center
    during the egress phase, based on the center of mass coordinates at
    two different contact times (t3 and t4). The shift depends on the
    ratio of planetary to stellar radius and the relative stellar radius
    at a given wavelength.

    Parameters
    ----------
    cbx_t3 : float or array-like
        X-coordinate of the center of mass at contact time t3
        (in stellar radii at the reference wavelength).
    cby_t3 : float or array-like
        Y-coordinate of the center of mass at contact time t3
        (in stellar radii at the reference wavelength).
    cbx_t4 : float or array-like
        X-coordinate of the center of mass at contact time t4
        (in stellar radii at the reference wavelength).
    cby_t4 : float or array-like
        Y-coordinate of the center of mass at contact time t4
        (in stellar radii at the reference wavelength).
    k : float or array-like
        Ratio of planetary radius to stellar radius at a given wavelength.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dcx : float or array-like
        X-coordinate of the center shift
        (in units of stellar radii at the reference wavelength).
    dcy : float or array-like
        Y-coordinate of the center shift
        (in units of stellar radii at the reference wavelength).
    """
    rs_alpha = np.asarray(rs_alpha)
    if rs_alpha.ndim != 0:
        rs_alpha = rs_alpha[None, :]
    mx = (cbx_t4 + cbx_t3) / 2
    my = (cby_t4 + cby_t3) / 2
    dx = (cbx_t4 - cbx_t3) / 2
    dy = (cby_t4 - cby_t3) / 2
    d = np.sqrt(dx**2 + dy**2)
    dx_n = dx / d
    dy_n = dy / d
    dcx = (
        -mx
        + dx_n * rs_alpha**2 * k / d
        + dy_n * jnp.sqrt((rs_alpha**2 - d**2) * (1 - rs_alpha**2 * (k / d) ** 2))
    )
    dcy = (
        -my
        + dy_n * rs_alpha**2 * k / d
        - dx_n * jnp.sqrt((rs_alpha**2 - d**2) * (1 - rs_alpha**2 * (k / d) ** 2))
    )
    # dcx and dcy are NaN when (1 - rs_alpha**2 * (k / d) ** 2) < 0
    # Replace NaN with min(dcx) and max(dcy)
    print("The number of NaN (egress)", len(np.where(np.isnan(dcx))[0]))
    print("The index of NaN (egress)", np.where(np.isnan(dcx)))
    dcx = np.nan_to_num(dcx, nan=np.nanmax(dcx))
    dcy = np.nan_to_num(dcy, nan=np.nanmax(dcy))
    return dcx, dcy


def delta_c_circular_orbit(
    period, a_over_rs_b, cosi_b, t0_b, k, Ttot, Tfull, t0, rs_alpha=1
):
    """
    Compute the center shift of a planetary shadow in a circular orbit
    during ingress and egress.

    This function calculates the center shift of a planetary shadow
    during both the ingress and egress phases for a planet in a circular
    orbit. The function takes into account the orbital parameters and
    the size ratio of the planet and star.

    Parameters
    ----------
    period : float
        Orbital period of the planet.
    a_over_rs_b : float
        Semi-major axis for the center of mass divided by stellar radius.
    cosi_b : float
        Cosine of the orbital inclination angle for the center of mass .
    t0_b : float
        Time of conjunction for the center of mass.
    k : float or array-like
        Ratio of planetary radius to stellar radius.
    Ttot : float or array-like
        Total transit duration.
    Tfull : float or array-like
        Full transit duration.
    t0 : float or array-like
        Midpoint of the transit (mid-transit time).
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    dcx_ingress : float or array-like
        X-coordinate of the center shift during ingress.
    dcy_ingress : float or array-like
        Y-coordinate of the center shift during ingress.
    dcx_egress : float or array-like
        X-coordinate of the center shift during egress.
    dcy_egress : float or array-like
        Y-coordinate of the center shift during egress.
    """
    # Contact times from Ttot, Tfull, and t0
    t1 = t0 - Ttot / 2
    t2 = t0 - Tfull / 2
    t3 = t0 + Tfull / 2
    t4 = t0 + Ttot / 2
    cbx_t1, cby_t1 = coordinates_circular_orbit(t1, period, a_over_rs_b, cosi_b, t0_b)
    cbx_t2, cby_t2 = coordinates_circular_orbit(t2, period, a_over_rs_b, cosi_b, t0_b)
    cbx_t3, cby_t3 = coordinates_circular_orbit(t3, period, a_over_rs_b, cosi_b, t0_b)
    cbx_t4, cby_t4 = coordinates_circular_orbit(t4, period, a_over_rs_b, cosi_b, t0_b)
    dcx_ingress, dcy_ingress = delta_c_ingress(
        cbx_t1, cby_t1, cbx_t2, cby_t2, k, rs_alpha
    )
    dcx_egress, dcy_egress = delta_c_egress(cbx_t3, cby_t3, cbx_t4, cby_t4, k, rs_alpha)
    return (
        dcx_ingress,
        dcy_ingress,
        dcx_egress,
        dcy_egress,
    )


def delta_c_circular_orbit_xiyixeye(
    period, a_over_rs_b, cosi_b, t0_b, k, Ttot, Tfull, t0, rs_alpha=1
):
    """
    Compute the center shift of a planetary shadow in transformed
    coordinate systems (xi, yi) or (xe, ye).

    This function calculates the center shift of a planetary shadow
    during both ingress and egress phases in transformed coordinate
    systems (xi, yi) or (xe, ye). The xi(e)-axis is aligned with the
    line connecting the stellar center to the point where the
    planetary center of mass overlaps the stellar edge.

    Parameters
    ----------
    period : float
        Orbital period of the planet.
    a_over_rs_b : float
        Semi-major axis  for the center of mass divided by stellar radius.
    cosi_b : float
        Cosine of the orbital inclination angle  for the center of mass.
    t0_b : float
        Time of conjunction for the center of mass.
    k : float or array-like
        Ratio of planetary radius to stellar radius.
    Ttot : float or array-like
        Total transit duration.
    Tfull : float or array-like
        Full transit duration (when the planet is fully overlapping the star).
    t0 : float or array-like
        Midpoint of the transit (mid-transit time).
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the
        reference wavelength (default is 1).

    Returns
    -------
    dci_xi : float or array-like
        xi-coordinate of the center shift during ingress.
    dci_yi : float or array-like
        yi-coordinate of the center shift during ingress.
    dce_xe : float or array-like
        xe-coordinate of the center shift during egress.
    dce_ye : float or array-like
        ye-coordinate of the center shift during egress.
    """
    # True anomaly when the center of mass intersects the edge of the star
    f_i = np.arcsin(np.sqrt(a_over_rs_b**2 - 1) / a_over_rs_b / np.sqrt(1 - cosi_b**2))
    f_e = np.pi - f_i
    cbX_tib = -a_over_rs_b * jnp.cos(f_i)
    cbY_tib = -a_over_rs_b * jnp.sin(f_i) * cosi_b
    cbX_teb = -a_over_rs_b * jnp.cos(f_e)
    cbY_teb = -a_over_rs_b * jnp.sin(f_e) * cosi_b

    dcx_ingress, dcy_ingress, dcx_egress, dcy_egress = delta_c_circular_orbit(
        period, a_over_rs_b, cosi_b, t0_b, k, Ttot, Tfull, t0, rs_alpha
    )
    dci_xi = -(dcx_ingress * cbX_tib + dcy_ingress * cbY_tib)
    dci_yi = -(-dcx_ingress * cbY_tib + dcy_ingress * cbX_tib)
    dce_xe = dcx_egress * cbX_teb + dcy_egress * cbY_teb
    dce_ye = -dcx_egress * cbY_teb + dcy_egress * cbX_teb
    return dci_xi, dci_yi, dce_xe, dce_ye


def rp_xip_xin_xep_xen(rp_over_rs, dci_xi, dce_xe, rs_alpha=1):
    """
    Calculate the rp spectra in both ingress and egress using the center shifts.

    This function computes the planetary radius at ingress and egress,
    taking into account the center shifts (`dci_xi` and `dce_xe`) and
    the stellar radius at a given wavelength (`rs_alpha`).

    Parameters
    ----------
    rp_over_rs : float or array-like
        Ratio of planetary radius to stellar radius.
    dci_xi : float or array-like
        xi-coordinate shift during ingress.
    dce_xe : float or array-like
        xe-coordinate shift during egress.
    rs_alpha : float or array-like, optional
        Ratio of stellar radius at the given wavelength to the reference
        wavelength (default is 1).

    Returns
    -------
    rp_xip : float or array-like
        Planetary radius in the positive xi direction during ingress.
    rp_xin : float or array-like
        Planetary radius in the negative xi direction during ingress.
    rp_xep : float or array-like
        Planetary radius in the positive xe direction during egress.
    rp_xen : float or array-like
        Planetary radius in the negative xe direction during egress.
    """
    rs_alpha = np.asarray(rs_alpha)
    if rs_alpha.ndim != 0:
        rs_alpha = rs_alpha[None, :]

    rp_xip = rs_alpha * rp_over_rs + dci_xi
    rp_xin = rs_alpha * rp_over_rs - dci_xi
    rp_xep = rs_alpha * rp_over_rs + dce_xe
    rp_xen = rs_alpha * rp_over_rs - dce_xe
    return rp_xip, rp_xin, rp_xep, rp_xen


if __name__ == "__main__":
    pass
