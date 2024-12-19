from wasp39b_params import period_day
from contact_times_hmc_plot import plot_all
from calc_light_curve import (
    transit_compute_flux_ecc0,
    transit_asymmetric_compute_flux,
)
from karate.calc_contact_times import calc_contact_times

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


if __name__ == "__main__":
    dir_output = "mcmc_results/dc_model_ecc0_jitter_0/"
    os.makedirs(dir_output, exist_ok=True)

    default_fontsize = plt.rcParams["font.size"]
    fontsize = 28
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    rng_key = random.key(0)
    rng_key, rng_key_ = random.split(rng_key)
    num_warmup = 1000
    num_samples = 1000

    wavelength = jnp.linspace(3.0, 5.0, 21)
    time = jnp.linspace(-150, 150, 301) * 65
    rp_over_rs = 0.150 + 0.005 * jnp.sin(wavelength * jnp.pi)
    t0 = 0
    period = period_day * 24 * 60 * 60
    a_over_rs = 11.4
    ecc = [[0], [0.1], [0.1]]
    omega = [[0], [0], [jnp.pi / 2]]
    cosi = 0.45 / a_over_rs
    u1 = 0.1
    u2 = 0.1
    jitter = 1.0 * 10 ** (-10)
    # jitter = 0.0005
    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )

    dc_over_rs_x_ingress = 0.005 * jnp.cos(wavelength * 1.2 * jnp.pi + 0.5 * jnp.pi)
    dc_over_rs_x_egress = 0.005 * jnp.cos(wavelength * 1.5 * jnp.pi + 0.1 * jnp.pi)
    dc_over_rs_y_ingress = 0.005 * jnp.sin(wavelength * 1.5 * jnp.pi - 0.2 * jnp.pi)
    dc_over_rs_y_egress = 0.005 * jnp.sin(wavelength * 1.2 * jnp.pi + jnp.pi)

    flux = transit_asymmetric_compute_flux(
        time,
        rp_over_rs,
        dc_over_rs_x_ingress,
        dc_over_rs_y_ingress,
        dc_over_rs_x_egress,
        dc_over_rs_y_egress,
        t0,
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        u1,
        u2,
    )
    error = jitter * random.normal(rng_key_, shape=flux.shape)
    # error = jitter * random.normal(rng_key_, shape=flux.shape)
    flux = flux + error
    jnp.save(dir_output + "flux", flux)

    #################### eccentricity = 0 fixed ####################

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
    input_values_dict["rp_over_rs"] = rp_over_rs
    input_values_dict["t0"] = t0
    input_values_dict["period"] = period
    input_values_dict["a_over_rs"] = a_over_rs
    input_values_dict["ecc"] = ecc
    input_values_dict["omega"] = omega
    input_values_dict["cosi"] = cosi
    input_values_dict["u1"] = u1
    input_values_dict["u2"] = u2
    input_values_dict["jitter"] = jitter
    input_values_dict["dc_over_rs_x_ingress"] = dc_over_rs_x_ingress
    input_values_dict["dc_over_rs_y_ingress"] = dc_over_rs_y_ingress
    input_values_dict["dc_over_rs_x_egress"] = dc_over_rs_x_egress
    input_values_dict["dc_over_rs_y_egress"] = dc_over_rs_y_egress

    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=False,
        deltac=True,
    )
