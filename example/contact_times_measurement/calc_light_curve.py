from karate.elements_to_orbit import orbital_elements_to_coordinates
from karate.elements_to_orbit import orbital_elements_to_coordinates_circular
from karate.calc_contact_times import expand_arrays
from jaxoplanet.core.limb_dark import light_curve

import jax.numpy as jnp
import jax
from jax import jit


@jit
def transit_compute_flux(
    time,
    rp_over_rs,
    t0,
    period,
    a_over_rs,
    ecc,
    omega,
    cosi,
    u1,
    u2,
    supersample_factor=10,
):
    """ """
    supersample_num = int(supersample_factor // 2 * 2 + 1)
    exposure_time = jnp.median(jnp.diff(time))
    dtarr = jnp.linspace(-0.5 * exposure_time, 0.5 * exposure_time, supersample_num)
    t_super = (time[:, None] + dtarr).ravel()

    rsky_over_rs_x, rsky_over_rs_y = orbital_elements_to_coordinates(
        t_super, period, a_over_rs, ecc, omega, cosi, t0
    )
    rsky_over_rs = jnp.sqrt(rsky_over_rs_x**2 + rsky_over_rs_y**2)

    flux_super = flux_from_rsky_over_rs(rsky_over_rs, rp_over_rs, u1, u2)
    flux = jnp.mean(
        flux_super.reshape(
            *flux_super.shape[:-1],
            flux_super.shape[-1] // supersample_num,
            supersample_num,
        ),
        axis=-1,
    )
    return flux


@jit
def transit_compute_flux_ecc0(
    time,
    rp_over_rs,
    t0,
    period,
    a_over_rs,
    cosi,
    u1,
    u2,
    supersample_factor=10,
):
    """ """
    supersample_num = int(supersample_factor // 2 * 2 + 1)
    exposure_time = jnp.median(jnp.diff(time))
    dtarr = jnp.linspace(-0.5 * exposure_time, 0.5 * exposure_time, supersample_num)
    t_super = (time[:, None] + dtarr).ravel()

    rsky_over_rs_x, rsky_over_rs_y = orbital_elements_to_coordinates_circular(
        t_super, period, a_over_rs, cosi, t0
    )
    rsky_over_rs = jnp.sqrt(rsky_over_rs_x**2 + rsky_over_rs_y**2)

    flux_super = flux_from_rsky_over_rs(rsky_over_rs, rp_over_rs, u1, u2)
    flux = jnp.mean(
        flux_super.reshape(
            *flux_super.shape[:-1],
            flux_super.shape[-1] // supersample_num,
            supersample_num,
        ),
        axis=-1,
    )
    return flux


@jit
def flux_from_rsky_over_rs(rsky_over_rs, rp_over_rs, u1, u2):
    inputs = [rp_over_rs, u1, u2]
    # Process each input: add a new axis if it is an array
    processed_inputs = []
    for inp in inputs:
        arr = jnp.asarray(inp)  # Convert to jax.numpy array
        if arr.ndim > 0:  # Check if it is an array (not a scalar)
            arr = jnp.expand_dims(arr, axis=-1)  # Add a new axis
        processed_inputs.append(arr)
    rp_over_rs, u1, u2 = processed_inputs

    rsky_over_rs, rp_over_rs, u1, u2 = expand_arrays(rsky_over_rs, rp_over_rs, u1, u2)
    u = jnp.stack((u1, u2), axis=-1)

    func = light_curve
    input_ndim = rsky_over_rs.ndim
    for _ in range(input_ndim):
        func = jax.vmap(func)
    # Process vector inputs
    flux = func(u, rsky_over_rs, rp_over_rs)
    return flux + 1.0
