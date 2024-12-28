from wasp39b_params import period_day
from contact_times_hmc_plot import plot_all
from calc_light_curve import transit_compute_flux, transit_compute_flux_ecc0

import catwoman
import numpy as np

import os
import sys
import matplotlib.pyplot as plt

from jax import random
from jax import config
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.distributions import constraints


# config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def model_ecc0(flux_obs, time, num_lightcurve):
    with numpyro.plate("n_light_curve", num_lightcurve, dim=-1):
        period = period_day * 24 * 60 * 60
        t0 = numpyro.sample("t0", dist.Uniform(-5000, 5000))
        Ttot = numpyro.sample("Ttot", dist.Uniform(5000, 15000))
        Tfull = numpyro.sample("Tfull", dist.Uniform(1000, Ttot))

        theta_tot = 2 * jnp.pi * Ttot / period
        theta_full = 2 * jnp.pi * Tfull / period
        depth_max = (
            (jnp.sin(theta_tot / 2) - jnp.sin(theta_full / 2))
            / (jnp.sin(theta_tot / 2) + jnp.sin(theta_full / 2))
        ) ** 2
        depth = numpyro.sample("depth", dist.Uniform(0, depth_max))

        u1 = numpyro.sample("u1", dist.Uniform(-3, 3))
        u2 = numpyro.sample("u2", dist.Uniform(-3, 3))
        baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01))
        jitter = numpyro.sample("jitter", dist.Uniform(0, 0.01))

    rp_over_rs = jnp.sqrt(depth)
    a_over_rs = jnp.sqrt(
        (
            -((1 - jnp.sqrt(depth)) ** 2) * jnp.cos(theta_tot / 2) ** 2
            + (1 + jnp.sqrt(depth)) ** 2 * jnp.cos(theta_full / 2) ** 2
        )
        / jnp.sin((theta_tot + theta_full) / 2)
        / jnp.sin((theta_tot - theta_full) / 2)
    )
    cosi = jnp.sqrt(
        (
            (1 - jnp.sqrt(depth)) ** 2 * jnp.sin(theta_tot / 2) ** 2
            - (1 + jnp.sqrt(depth)) ** 2 * jnp.sin(theta_full / 2) ** 2
        )
        / (
            -((1 - jnp.sqrt(depth)) ** 2) * jnp.cos(theta_tot / 2) ** 2
            + (1 + jnp.sqrt(depth)) ** 2 * jnp.cos(theta_full / 2) ** 2
        )
    )
    flux = transit_compute_flux_ecc0(
        time, rp_over_rs, t0, period, a_over_rs, cosi, u1, u2
    )
    with numpyro.plate("wavelength", num_lightcurve, dim=-2):
        with numpyro.plate("time", len(time), dim=-1):
            numpyro.sample(
                "light_curve",
                dist.Normal(
                    flux * baseline[:, None],
                    jitter[:, None] * jnp.ones_like(flux),
                ),
                obs=flux_obs,
            )


def guide_ecc0(flux_obs, time, num_lightcurve):
    with numpyro.plate("n_light_curve", num_lightcurve, dim=-1):
        loc_t0 = numpyro.param("loc_t0", 0.0 * jnp.ones(num_lightcurve))
        scale_t0 = numpyro.param(
            "scale_t0", 10.0 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample(
            "t0", dist.TruncatedNormal(loc_t0, scale_t0, low=-5000, high=5000)
        )

        loc_Ttot = numpyro.param(
            "loc_Ttot",
            10000.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_Ttot = numpyro.param(
            "scale_Ttot",
            50.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        Ttot = numpyro.sample(
            "Ttot", dist.TruncatedNormal(loc_Ttot, scale_Ttot, low=9000, high=13000)
        )

        loc_Tfull = numpyro.param(
            "loc_Tfull",
            7000.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_Tfull = numpyro.param(
            "scale_Tfull",
            50.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        Tfull = numpyro.sample(
            "Tfull",
            dist.TruncatedNormal(
                loc_Tfull, scale_Tfull, low=6000 * jnp.ones(num_lightcurve), high=Ttot
            ),
        )

        theta_tot = 2 * jnp.pi * Ttot / period
        theta_full = 2 * jnp.pi * Tfull / period
        depth_max = (
            (jnp.sin(theta_tot / 2) - jnp.sin(theta_full / 2))
            / (jnp.sin(theta_tot / 2) + jnp.sin(theta_full / 2))
        ) ** 2

        loc_depth = numpyro.param(
            "loc_depth",
            0.02 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_depth = numpyro.param(
            "scale_depth",
            0.0001 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "depth",
            dist.TruncatedNormal(
                loc_depth,
                scale_depth,
                low=0.01 * jnp.ones(num_lightcurve),
                high=depth_max,
            ),
        )

        loc_u1 = numpyro.param("loc_u1", 0.1 * jnp.ones(num_lightcurve))
        scale_u1 = numpyro.param(
            "scale_u1", 0.2 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample("u1", dist.TruncatedNormal(loc_u1, scale_u1, low=-3, high=3))

        loc_u2 = numpyro.param("loc_u2", 0.1 * jnp.ones(num_lightcurve))
        scale_u2 = numpyro.param(
            "scale_u2", 0.2 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample("u2", dist.TruncatedNormal(loc_u2, scale_u2, low=-3, high=3))

        loc_baseline = numpyro.param(
            "loc_baseline",
            1.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_baseline = numpyro.param(
            "scale_baseline",
            0.005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "baseline",
            dist.TruncatedNormal(loc_baseline, scale_baseline, low=0.99, high=1.01),
        )

        loc_jitter = numpyro.param(
            "loc_jitter",
            0.005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_jitter = numpyro.param(
            "scale_jitter",
            0.0005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "jitter", dist.TruncatedNormal(loc_jitter, scale_jitter, low=0, high=0.01)
        )


def model(flux_obs, time, num_lightcurve):
    period = period_day * 24 * 60 * 60
    with numpyro.plate("n_light_curve", num_lightcurve, dim=-1):
        rp_over_rs = numpyro.sample("rp_over_rs", dist.Uniform(0, 0.5))
        t0 = numpyro.sample("t0", dist.Uniform(-5000, 5000))
        a_over_rs = numpyro.sample("a_over_rs", dist.Uniform(1, 50))
        ecosw = numpyro.sample("ecosw", dist.Uniform(-1.0, 1.0))
        esinw = numpyro.sample("esinw", dist.Uniform(-1.0, 1.0))
        b = numpyro.sample("b", dist.Uniform(0 * rp_over_rs, 1.0 - rp_over_rs))
        u1 = numpyro.sample("u1", dist.Uniform(-3, 3))
        u2 = numpyro.sample("u2", dist.Uniform(-3, 3))
        baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01))
        jitter = numpyro.sample("jitter", dist.Uniform(0, 0.01))

    cosi = b / a_over_rs
    ecc = jnp.sqrt(ecosw**2 + esinw**2)
    omega = jnp.arctan2(esinw, ecosw)
    flux = transit_compute_flux(
        time, rp_over_rs, t0, period, a_over_rs, ecc, omega, cosi, u1, u2
    )
    with numpyro.plate("wavelength", num_lightcurve, dim=-2):
        with numpyro.plate("time", len(time), dim=-1):
            numpyro.sample(
                "light_curve",
                dist.Normal(
                    flux * baseline[:, None],
                    jitter[:, None] * jnp.ones_like(flux),
                ),
                obs=flux_obs,
            )


def guide(flux_obs, time, num_lightcurve):
    with numpyro.plate("n_light_curve", num_lightcurve, dim=-1):
        loc_rp_over_rs = numpyro.param(
            "loc_rp_over_rs", 0.145 * jnp.ones(num_lightcurve)
        )
        scale_rp_over_rs = numpyro.param(
            "scale_rp_over_rs",
            0.002 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        rp_over_rs = numpyro.sample(
            "rp_over_rs",
            dist.TruncatedNormal(loc_rp_over_rs, scale_rp_over_rs, low=0, high=0.5),
        )

        loc_t0 = numpyro.param("loc_t0", 0.0 * jnp.ones(num_lightcurve))
        scale_t0 = numpyro.param(
            "scale_t0", 10.0 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample(
            "t0", dist.TruncatedNormal(loc_t0, scale_t0, low=-5000, high=5000)
        )

        loc_a_over_rs = numpyro.param("loc_a_over_rs", 11.4 * jnp.ones(num_lightcurve))
        scale_a_over_rs = numpyro.param(
            "scale_a_over_rs",
            0.1 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "a_over_rs",
            dist.TruncatedNormal(loc_a_over_rs, scale_a_over_rs, low=5, high=20),
        )

        loc_ecosw = numpyro.param("loc_ecosw", 0.05 * jnp.ones(num_lightcurve))
        scale_ecosw = numpyro.param(
            "scale_ecosw",
            0.1 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "ecosw", dist.TruncatedNormal(loc_ecosw, scale_ecosw, low=-1.0, high=1.0)
        )

        loc_esinw = numpyro.param("loc_esinw", 0.05 * jnp.ones(num_lightcurve))
        scale_esinw = numpyro.param(
            "scale_esinw",
            0.1 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "esinw", dist.TruncatedNormal(loc_esinw, scale_esinw, low=-1.0, high=1.0)
        )

        loc_b = numpyro.param("loc_b", 0.45 * jnp.ones(num_lightcurve))
        scale_b = numpyro.param(
            "scale_b",
            0.05 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "b",
            dist.TruncatedNormal(
                loc_b, scale_b, low=0 * rp_over_rs, high=1.0 - rp_over_rs
            ),
        )

        loc_u1 = numpyro.param("loc_u1", 0.1 * jnp.ones(num_lightcurve))
        scale_u1 = numpyro.param(
            "scale_u1", 0.2 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample("u1", dist.TruncatedNormal(loc_u1, scale_u1, low=-3, high=3))

        loc_u2 = numpyro.param("loc_u2", 0.1 * jnp.ones(num_lightcurve))
        scale_u2 = numpyro.param(
            "scale_u2", 0.2 * jnp.ones(num_lightcurve), constraint=constraints.positive
        )
        numpyro.sample("u2", dist.TruncatedNormal(loc_u2, scale_u2, low=-3, high=3))

        loc_baseline = numpyro.param(
            "loc_baseline",
            1.0 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_baseline = numpyro.param(
            "scale_baseline",
            0.005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "baseline",
            dist.TruncatedNormal(loc_baseline, scale_baseline, low=0.99, high=1.01),
        )

        loc_jitter = numpyro.param(
            "loc_jitter",
            0.005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        scale_jitter = numpyro.param(
            "scale_jitter",
            0.0005 * jnp.ones(num_lightcurve),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "jitter", dist.TruncatedNormal(loc_jitter, scale_jitter, low=0, high=0.01)
        )


if __name__ == "__main__":
    dir_list = [
        "mcmc_results/catwoman_model_ecc0_jitter_0/",
        "mcmc_results/catwoman_model_eccfree_jitter_0/",
    ]

    default_fontsize = plt.rcParams["font.size"]
    fontsize = 28
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"  # LaTeXに近いスタイル
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    rng_key = random.key(0)
    rng_key, rng_key_ = random.split(rng_key)
    num_warmup = 1000
    num_samples = 1000

    wavelength = np.linspace(3.0, 5.0, 21)
    time = np.linspace(-150, 150, 301) * 65
    rp_over_rs_evening = 0.150 + 0.005 * np.sin(wavelength * np.pi)
    rp_over_rs_morning = 0.150 + 0.005 * np.sin(wavelength * np.pi * 1.6 + np.pi * 0.5)
    t0 = 0
    period = period_day * 24 * 60 * 60
    a_over_rs = 11.4
    ecc = [[0], [0.1], [0.1]]
    omega = [[0], [0], [np.pi / 2]]
    cosi = 0.45 / a_over_rs
    u1 = 0.1
    u2 = 0.1
    # jitter = 0.00025
    jitter = 0

    flux = np.zeros((len(ecc), len(rp_over_rs_evening), len(time)))
    params = catwoman.TransitParams()  # object to store transit parameters
    params.t0 = t0  # time of inferior conjuction (in days)
    params.per = period  # orbital period (in days)
    params.a = a_over_rs  # semi-major axis (in units of stellar radii)
    params.inc = np.arccos(cosi) / np.pi * 180.0  # orbital inclination (in degrees)
    params.u = [u1, u2]  # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # limb darkening model
    params.phi = 90.0  # angle of rotation of top semi-circle
    for i in range(len(ecc)):
        params.ecc = np.array(ecc)[i, 0]  # eccentricity
        params.w = (
            np.array(omega)[i, 0] / np.pi * 180.0
        )  # longitude of periastron (in degrees)
        for j in range(len(rp_over_rs_evening)):
            params.rp = rp_over_rs_evening[
                j
            ]  # top semi-circle radius (in units of stellar radii)
            params.rp2 = rp_over_rs_morning[
                j
            ]  # bottom semi-circle radius (in units of stellar radii)

            model_catwoman = catwoman.TransitModel(params, time)  # initalises model
            flux[i, j] = model_catwoman.light_curve(params)  # calculates light curve

    error = jitter * random.normal(rng_key_, shape=flux.shape)
    flux = flux + error

    #################### eccentricity = 0 fixed ####################
    dir_output = dir_list[0]
    os.makedirs(dir_output, exist_ok=True)
    jnp.save(dir_output + "flux", flux)

    # SVI
    rng_key, rng_key_ = random.split(rng_key)
    optimizer = numpyro.optim.Adam(step_size=0.0005)
    svi = SVI(model_ecc0, guide_ecc0, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key_,
        2000,
        flux_obs=flux.reshape((-1, len(time))),
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    params_svi = svi_result.params

    # MCMC initial values
    params = ["t0", "Ttot", "Tfull", "depth", "u1", "u2", "baseline", "jitter"]
    init_values = {}
    for param in params:
        print("loc_" + param, params_svi["loc_" + param])
        print("scale_" + param, params_svi["scale_" + param])
        init_values[param] = params_svi["loc_" + param]
    init_strategy = init_to_value(values=init_values)

    # MCMC
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(
        model_ecc0,
        forward_mode_differentiation=False,
        # max_tree_depth=13,
        init_strategy=init_strategy,
        # target_accept_prob=0.9,
    )
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        flux_obs=flux.reshape((-1, len(time))),
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )

    mcmc.print_summary()
    with open(dir_output + "mcmc_summary.txt", "w") as f:
        # save current stdout
        original_stdout = sys.stdout
        # redirect stdout to a file
        sys.stdout = f

        # write output to file
        mcmc.print_summary()

        # restore the stdout
        sys.stdout = original_stdout

    # Get samples
    posterior_sample = mcmc.get_samples()
    jnp.savez(dir_output + "posterior_sample", **posterior_sample)

    # Prediction
    rng_key, rng_key_ = random.split(rng_key)
    pred = Predictive(model_ecc0, posterior_sample)
    predictions = pred(
        rng_key_,
        flux_obs=None,
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    jnp.savez(dir_output + "predictions", **predictions)

    # Plot
    input_values_dict = {}
    input_values_dict["wavelength"] = wavelength
    input_values_dict["time"] = time
    input_values_dict["rp_over_rs_morning"] = rp_over_rs_morning
    input_values_dict["rp_over_rs_evening"] = rp_over_rs_evening
    input_values_dict["t0"] = t0
    input_values_dict["period"] = period
    input_values_dict["a_over_rs"] = a_over_rs
    input_values_dict["ecc"] = ecc
    input_values_dict["omega"] = omega
    input_values_dict["cosi"] = cosi
    input_values_dict["u1"] = u1
    input_values_dict["u2"] = u2
    input_values_dict["jitter"] = jitter

    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=False,
        deltac=False,
        catwoman=True,
    )

    #################### eccentricity free ####################
    dir_output = dir_list[1]
    os.makedirs(dir_output, exist_ok=True)
    jnp.save(dir_output + "flux", flux)

    # SVI
    rng_key, rng_key_ = random.split(rng_key)
    optimizer = numpyro.optim.Adam(step_size=0.0005)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key_,
        2000,
        flux_obs=flux.reshape((-1, len(time))),
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    params_svi = svi_result.params

    # MCMC initial values
    params = [
        "rp_over_rs",
        "t0",
        "a_over_rs",
        "ecosw",
        "esinw",
        "b",
        "u1",
        "u2",
        "baseline",
        "jitter",
    ]
    init_values = {}
    for param in params:
        print("loc_" + param, params_svi["loc_" + param])
        print("scale_" + param, params_svi["scale_" + param])
        init_values[param] = params_svi["loc_" + param]
    init_strategy = init_to_value(values=init_values)

    # MCMC
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(
        model,
        forward_mode_differentiation=False,
        # max_tree_depth=12,
        init_strategy=init_strategy,
        # target_accept_prob=0.9,
    )
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        flux_obs=flux.reshape((-1, len(time))),
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )

    mcmc.print_summary()
    with open(dir_output + "mcmc_summary.txt", "w") as f:
        # save current stdout
        original_stdout = sys.stdout
        # redirect stdout to a file
        sys.stdout = f

        # write output to file
        mcmc.print_summary()

        # restore the stdout
        sys.stdout = original_stdout

    # Get samples
    posterior_sample = mcmc.get_samples()
    jnp.savez(dir_output + "posterior_sample", **posterior_sample)

    # Prediction
    rng_key, rng_key_ = random.split(rng_key)
    pred = Predictive(model, posterior_sample)
    predictions = pred(
        rng_key_,
        flux_obs=None,
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    jnp.savez(dir_output + "predictions", **predictions)

    # Plot
    input_values_dict = {}
    input_values_dict["wavelength"] = wavelength
    input_values_dict["time"] = time
    input_values_dict["rp_over_rs_morning"] = rp_over_rs_morning
    input_values_dict["rp_over_rs_evening"] = rp_over_rs_evening
    input_values_dict["t0"] = t0
    input_values_dict["period"] = period
    input_values_dict["a_over_rs"] = a_over_rs
    input_values_dict["ecc"] = ecc
    input_values_dict["omega"] = omega
    input_values_dict["cosi"] = cosi
    input_values_dict["u1"] = u1
    input_values_dict["u2"] = u2
    input_values_dict["jitter"] = jitter

    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=True,
        deltac=False,
        catwoman=True,
    )
