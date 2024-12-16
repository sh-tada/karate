import pytest
import jax.numpy as jnp
from jax import config
from karate.calc_contact_times import (
    calc_contact_times,
    calc_contact_times_circular,
)

# Ensure JAX outputs are deterministic for tests
config.update("jax_enable_x64", True)


def test_calc_contact_times_circular():
    rp_over_rs = 0.1
    period = 10
    a_over_rs = 10
    cosi = 0.01
    t0 = 0.0
    t1, t2, t3, t4 = calc_contact_times_circular(
        rp_over_rs, period, a_over_rs, cosi, t0
    )
    assert jnp.allclose(t2 - t1, t4 - t3, atol=1e-8)

    rp_over_rs = jnp.array([0.1, 0.11])
    period = 10
    a_over_rs = jnp.array([[10, 11], [12, 13], [14, 15]])
    cosi = jnp.array([[0.01], [0.011], [0.012]])
    t0 = jnp.array([0.0, 0.1])
    t1, t2, t3, t4 = calc_contact_times_circular(
        rp_over_rs, period, a_over_rs, cosi, t0
    )
    assert jnp.allclose(t2 - t1, t4 - t3, atol=1e-8)

    theta_tot = (t4 - t1) / period * 2 * jnp.pi
    theta_full = (t3 - t2) / period * 2 * jnp.pi
    a_over_rs_fromT = jnp.sqrt(
        (
            -((1 - rp_over_rs) ** 2) * jnp.cos(theta_tot / 2) ** 2
            + (1 + rp_over_rs) ** 2 * jnp.cos(theta_full / 2) ** 2
        )
        / jnp.sin((theta_tot + theta_full) / 2)
        / jnp.sin((theta_tot - theta_full) / 2)
    )
    cosi_fromT = jnp.sqrt(
        (
            (1 - rp_over_rs) ** 2 * jnp.sin(theta_tot / 2) ** 2
            - (1 + rp_over_rs) ** 2 * jnp.sin(theta_full / 2) ** 2
        )
        / (
            -((1 - rp_over_rs) ** 2) * jnp.cos(theta_tot / 2) ** 2
            + (1 + rp_over_rs) ** 2 * jnp.cos(theta_full / 2) ** 2
        )
    )
    assert jnp.allclose(a_over_rs, a_over_rs_fromT, atol=1e-8)
    assert jnp.allclose(cosi, cosi_fromT, atol=1e-8)


def test_calc_contact_times():
    rp_over_rs = 0.11
    period = 11
    a_over_rs = 11
    ecc = 0
    omega = 0
    cosi = 0.011
    t0 = 0.1
    t1_c, t2_c, t3_c, t4_c = calc_contact_times_circular(
        rp_over_rs, period, a_over_rs, cosi, t0
    )
    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )

    assert jnp.allclose(t1_c, t1, atol=1e-8)
    assert jnp.allclose(t2_c, t2, atol=1e-8)
    assert jnp.allclose(t3_c, t3, atol=1e-8)
    assert jnp.allclose(t4_c, t4, atol=1e-8)

    rp_over_rs = jnp.array([0.1, 0.11])
    period = 10
    a_over_rs = jnp.array([[10, 11], [12, 13], [14, 15]])
    ecc = jnp.array([0, 0])
    omega = jnp.array([0, 0])
    cosi = jnp.array([[0.01], [0.011], [0.012]])
    t0 = jnp.array([0.0, 0.1])
    t1_c, t2_c, t3_c, t4_c = calc_contact_times_circular(
        rp_over_rs, period, a_over_rs, cosi, t0
    )
    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )

    assert jnp.allclose(t1_c, t1, atol=1e-8)
    assert jnp.allclose(t2_c, t2, atol=1e-8)
    assert jnp.allclose(t3_c, t3, atol=1e-8)
    assert jnp.allclose(t4_c, t4, atol=1e-8)

    t0 = jnp.array([0.0, 0.1]) + 1.0

    t1_2, t2_2, t3_2, t4_2 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )
    assert jnp.allclose(t1_2, t1 + 1.0, atol=1e-8)
    assert jnp.allclose(t2_2, t2 + 1.0, atol=1e-8)
    assert jnp.allclose(t3_2, t3 + 1.0, atol=1e-8)
    assert jnp.allclose(t4_2, t4 + 1.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main()
