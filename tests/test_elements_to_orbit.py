import pytest
import jax.numpy as jnp
from jax import config
from atmoctv.elements_to_orbit import (
    true_anomaly_to_eccentric_anomaly,
    eccentric_anomaly_to_true_anomaly,
    eccentric_anomaly_to_t_from_tperi,
    tperi_to_t0,
    t0_to_tperi,
    orbital_elements_to_coordinates_circular,
    orbital_elements_to_coordinates,
)

# Ensure JAX outputs are deterministic for tests
config.update("jax_enable_x64", True)


def test_true_anomaly_to_eccentric_anomaly():
    f = jnp.pi / 3  # 60 degrees
    ecc = 0
    result = true_anomaly_to_eccentric_anomaly(f, ecc)
    assert jnp.allclose(result, f, atol=1e-8)

    f = jnp.pi / 3 + jnp.linspace(-jnp.pi, jnp.pi, 10)
    ecc = 0.1 + jnp.linspace(-0.1, 0.1, 10)
    result = true_anomaly_to_eccentric_anomaly(f, ecc)
    expected = 2 * jnp.arctan(jnp.sqrt((1 - ecc) / (1 + ecc)) * jnp.tan(f / 2))
    assert jnp.allclose(result, expected, atol=1e-8)


def test_eccentric_anomaly_to_true_anomaly():
    u = jnp.pi / 3  # 60 degrees
    ecc = 0
    result = eccentric_anomaly_to_true_anomaly(u, ecc)
    assert jnp.allclose(result, u, atol=1e-8)

    u = jnp.pi / 3 + jnp.linspace(-jnp.pi, jnp.pi, 10)
    ecc = 0.1 + jnp.linspace(-0.1, 0.1, 10)
    result = eccentric_anomaly_to_true_anomaly(u, ecc)
    expected = 2 * jnp.arctan(jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(u / 2))
    assert jnp.allclose(result, expected, atol=1e-8)

    result = true_anomaly_to_eccentric_anomaly(result, ecc)
    expected = (u + jnp.pi) % (2 * jnp.pi) - jnp.pi
    assert jnp.allclose(result, expected, atol=1e-8)


def test_eccentric_anomaly_to_t_from_tperi():
    u = jnp.pi / 4  # 45 degrees
    ecc = 0
    period = 10.0
    result = eccentric_anomaly_to_t_from_tperi(u, ecc, period)
    expected = period / (2 * jnp.pi) * u
    assert jnp.allclose(result, expected, atol=1e-8)

    u = jnp.pi / 4 + jnp.linspace(-jnp.pi, jnp.pi, 10)  # 45 degrees
    ecc = 0.2 + jnp.linspace(-0.1, 0.1, 10)
    period = 10.0 + jnp.linspace(-0.1, 0.1, 10)
    result = eccentric_anomaly_to_t_from_tperi(u, ecc, period)
    expected = period / (2 * jnp.pi) * (u - ecc * jnp.sin(u))
    assert jnp.allclose(result, expected, atol=1e-8)


def test_t0_to_tperi():
    t0 = 0.1
    period = 10.0
    ecc = 0
    omega = 0
    result = t0_to_tperi(t0, period, ecc, omega)
    assert jnp.allclose(result, -10.0 / 4.0 + t0, atol=1e-8)

    t0 = 0.0 + jnp.linspace(-0.1, 0.1, 10)
    period = 10.0 + jnp.linspace(-0.1, 0.1, 10)
    ecc = 0.2 + jnp.linspace(-0.1, 0.1, 10)
    omega = jnp.pi / 4 + jnp.linspace(-jnp.pi, jnp.pi, 10)
    result = t0_to_tperi(t0, period, ecc, omega)
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2.0 - omega, ecc)
    expected = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    assert jnp.allclose(result, t0 - expected, atol=1e-8)


def test_tperi_to_t0():
    t_periastron = 0.1
    period = 10.0
    ecc = 0
    omega = 0
    result = tperi_to_t0(t_periastron, period, ecc, omega)
    assert jnp.allclose(result, 10.0 / 4.0 + t_periastron, atol=1e-8)

    t_periastron = 0.0 + jnp.linspace(-0.1, 0.1, 10)
    period = 10.0 + jnp.linspace(-0.1, 0.1, 10)
    ecc = 0.2 + jnp.linspace(-0.1, 0.1, 10)
    omega = jnp.pi / 4 + jnp.linspace(-jnp.pi, jnp.pi, 10)
    result = tperi_to_t0(t_periastron, period, ecc, omega)
    u_t0 = true_anomaly_to_eccentric_anomaly(jnp.pi / 2 - omega, ecc)
    expected = eccentric_anomaly_to_t_from_tperi(u_t0, ecc, period)
    assert jnp.allclose(result, t_periastron + expected, atol=1e-8)


def test_orbital_elements_to_coordinates_circular():
    t = 0.0
    period = 10.0
    a_over_rs = 10.0
    cosi = 0.5
    t0 = 0.0
    x, y = orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0)

    expected_x = 0
    expected_y = -a_over_rs * cosi

    assert jnp.allclose(x, expected_x, atol=1e-8)
    assert jnp.allclose(y, expected_y, atol=1e-8)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
    period = 10.0
    a_over_rs = 10.0
    cosi = 0.01
    t0 = 0.0
    x, y = orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0)

    expected_x = a_over_rs * jnp.sin(2.0 * jnp.pi * (t - t0) / period)
    expected_y = -a_over_rs * jnp.cos(2.0 * jnp.pi * (t - t0) / period) * cosi

    assert jnp.allclose(x, expected_x, atol=1e-8)
    assert jnp.allclose(y, expected_y, atol=1e-8)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0]) + 1.0
    period = 10.0
    a_over_rs = 10.0
    cosi = 0.01
    t0 = 1.0
    x2, y2 = orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0)

    assert jnp.allclose(x, x2, atol=1e-8)
    assert jnp.allclose(y, y2, atol=1e-8)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
    period = jnp.array([10.0, 20.0])
    a_over_rs = jnp.array([10.0, 20.0])
    cosi = jnp.array([0.0, 0.01])
    t0 = jnp.array([0.0, 1.0])
    x, y = orbital_elements_to_coordinates_circular(t, period, a_over_rs, cosi, t0)

    expected_x = a_over_rs[:, None] * jnp.sin(
        2.0 * jnp.pi * (t - t0[:, None]) / period[:, None]
    )
    expected_y = (
        -a_over_rs[:, None]
        * jnp.cos(2.0 * jnp.pi * (t - t0[:, None]) / period[:, None])
        * cosi[:, None]
    )

    assert jnp.allclose(x, expected_x, atol=1e-8)
    assert jnp.allclose(y, expected_y, atol=1e-8)


def test_orbital_elements_to_coordinates():
    t = 1.0
    period = 10.0
    a_over_rs = 10.0
    ecc = 0
    omega = 0
    cosi = 0.5
    t0 = 0.0
    x, y = orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0)
    expected_x, expected_y = orbital_elements_to_coordinates_circular(
        t, period, a_over_rs, cosi, t0
    )
    assert jnp.allclose(x, expected_x, atol=1e-8)
    assert jnp.allclose(y, expected_y, atol=1e-8)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
    period = 10.0
    a_over_rs = 10.0
    ecc = 0.1
    omega = jnp.pi / 4
    cosi = 0.5
    t0 = 0.0
    x, y = orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0]) + 1.0
    t0 = 1.0
    x2, y2 = orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0)

    assert jnp.allclose(x, x2, atol=1e-8)
    assert jnp.allclose(y, y2, atol=1e-8)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
    period = jnp.array([10.0, 20.0])
    a_over_rs = jnp.array([10.0, 20.0])
    ecc = jnp.array([0.01, 0.01])
    omega = jnp.array([jnp.pi / 4, jnp.pi / 3])
    cosi = jnp.array([0.0, 0.01])
    t0 = jnp.array([0.0, 1.0])

    x, y = orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0)

    t = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0]) + 1.0
    t0 = jnp.array([0.0, 1.0]) + 1.0
    x2, y2 = orbital_elements_to_coordinates(t, period, a_over_rs, ecc, omega, cosi, t0)

    assert jnp.allclose(x, x2, atol=1e-8)
    assert jnp.allclose(y, y2, atol=1e-8)


if __name__ == "__main__":
    pytest.main()
