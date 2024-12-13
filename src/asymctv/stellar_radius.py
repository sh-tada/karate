from asymctv.elements_to_orbit import true_anomaly_to_eccentric_anomaly
from asymctv.calc_contact_times import calc_true_anomaly_rsky
import jax.numpy as jnp
from jax import jit


@jit
def duration_from_rs_alpha(
    rs_alpha,
    period,
    a_over_rs,
    ecc,
    omega,
    cosi,
):
    """ """
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


if __name__ == "__main__":
    pass
