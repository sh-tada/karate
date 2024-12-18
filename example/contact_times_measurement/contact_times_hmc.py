from wasp39b_params import period_day
from contact_times_hmc_plot import lightcurve_fit_plot, prediction_plot
from calc_light_curve import transit_compute_flux, transit_compute_flux_ecc0
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
from numpyro.diagnostics import hpdi
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

    rp_over_rs = numpyro.deterministic("rp_over_rs", jnp.sqrt(depth))
    a_over_rs = numpyro.deterministic(
        "a_over_rs",
        jnp.sqrt(
            (
                -((1 - jnp.sqrt(depth)) ** 2) * jnp.cos(theta_tot / 2) ** 2
                + (1 + jnp.sqrt(depth)) ** 2 * jnp.cos(theta_full / 2) ** 2
            )
            / jnp.sin((theta_tot + theta_full) / 2)
            / jnp.sin((theta_tot - theta_full) / 2)
        ),
    )
    cosi = numpyro.deterministic(
        "cosi",
        jnp.sqrt(
            (
                (1 - jnp.sqrt(depth)) ** 2 * jnp.sin(theta_tot / 2) ** 2
                - (1 + jnp.sqrt(depth)) ** 2 * jnp.sin(theta_full / 2) ** 2
            )
            / (
                -((1 - jnp.sqrt(depth)) ** 2) * jnp.cos(theta_tot / 2) ** 2
                + (1 + jnp.sqrt(depth)) ** 2 * jnp.cos(theta_full / 2) ** 2
            )
        ),
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
        t0 = numpyro.sample("t0_1d", dist.Uniform(-5000, 5000))
        a_over_rs = numpyro.sample("a_over_rs", dist.Uniform(1, 50))
        ecosw = numpyro.sample("ecosw", dist.Uniform(-1.0, 1.0))
        esinw = numpyro.sample("esinw", dist.Uniform(-1.0, 1.0))
        b = numpyro.sample("b", dist.Uniform(0, 1.0))
        u1 = numpyro.sample("u1", dist.Uniform(-3, 3))
        u2 = numpyro.sample("u2", dist.Uniform(-3, 3))
        baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01))
        jitter = numpyro.sample("jitter", dist.Uniform(0, 0.01))

    cosi = numpyro.deterministic("cosi", b / a_over_rs)
    ecc = numpyro.deterministic("ecc", jnp.sqrt(ecosw**2 + esinw**2))
    omega = numpyro.deterministic(
        "omega", jnp.where(ecc > 0, jnp.arctan2(esinw / ecc, ecosw / ecc), 0.0)
    )
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
        numpyro.sample(
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
        numpyro.sample("b", dist.TruncatedNormal(loc_b, scale_b, low=0, high=1.0))

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
    dir_output_ecc0 = "mcmc_results/model_ecc0_jitter_00001/"
    dir_output = "mcmc_results/model_eccfree_jitter_00001/"
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_output_ecc0, exist_ok=True)

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
    jitter = 0.0001
    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )

    flux = transit_compute_flux(
        time, rp_over_rs, t0, period, a_over_rs, ecc, omega, cosi, u1, u2
    )
    error = jitter * random.normal(rng_key_, shape=flux.shape)
    # error = jitter * random.normal(rng_key_, shape=flux.shape)
    flux = flux + error
    jnp.save(dir_output_ecc0 + "flux", flux)
    jnp.save(dir_output + "flux", flux)

    for i in range(len(flux)):
        fig = plt.figure(figsize=(12, 18))
        for j in range(len(flux[0])):
            plt.plot(
                time,
                flux[i, j] - 0.01 * j,
                marker=".",
                linestyle="None",
                label="$R_{p}/R_{s}$" + f" = {rp_over_rs[j]:.3f}",
            )
        # plt.legend()
        plt.title(
            f"Light Curve ($e\cos\omega$ ={ecc[i][0]*jnp.cos(omega[i][0]):.1f}, "
            + f"$e\sin\omega$ ={ecc[i][0]*jnp.sin(omega[i][0]):.1f})"
        )
        plt.xlabel("Time")
        plt.ylabel("Relative Flux")
        plt.savefig(
            f"lightcurve_ecosw{ecc[i][0]*jnp.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*jnp.sin(omega[i][0]):.1f}.png"
        )
        plt.close()

    num_warmup = 100
    num_samples = 100

    #################### eccentricity = 0 fixed ####################
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

    params = ["t0", "Ttot", "Tfull", "depth", "u1", "u2", "baseline", "jitter"]
    init_values = {}
    for param in params:
        print("loc_" + param, params_svi["loc_" + param])
        print("scale_" + param, params_svi["scale_" + param])
        init_values[param] = params_svi["loc_" + param]
    init_strategy = init_to_value(values=init_values)

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
    with open(dir_output_ecc0 + "mcmc_summary_ecc0.txt", "w") as f:
        # save current stdout
        original_stdout = sys.stdout
        # redirect stdout to a file
        sys.stdout = f

        # write output to file
        mcmc.print_summary()

        # restore the stdout
        sys.stdout = original_stdout

    # SAMPLING
    posterior_sample = mcmc.get_samples()
    jnp.savez(dir_output_ecc0 + "posterior_sample_ecc0", **posterior_sample)

    rng_key, rng_key_ = random.split(rng_key)
    pred = Predictive(model_ecc0, posterior_sample)
    predictions = pred(
        rng_key_,
        flux_obs=None,
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    jnp.savez(dir_output_ecc0 + "predictions_ecc0", **predictions)
    # predictions = jnp.load(dir_output_ecc0 + "predictions_ecc0.npz")

    pred_light_curve = predictions["light_curve"].reshape((num_samples, *flux.shape))
    for i in range(len(ecc)):
        pred_median = jnp.median(pred_light_curve[:, i], axis=0)
        pred_hpdi = hpdi(pred_light_curve[:, i], 0.90)
        lightcurve_fit_plot(
            time,
            flux[i],
            pred_median,
            pred_hpdi,
            hpdi_range=0.90,
            title=f"Light Curve ($e\cos\omega$ ={ecc[i][0]*jnp.cos(omega[i][0]):.1f}, "
            + f"$e\sin\omega$ ={ecc[i][0]*jnp.sin(omega[i][0]):.1f})",
            dir_output=dir_output_ecc0,
            filename=f"fit_lightcurve_ecosw{ecc[i][0]*jnp.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*jnp.sin(omega[i][0]):.1f}_ecc0fix.png",
        )

    # posterior_sample = jnp.load(dir_output_ecc0 + "posterior_sample_ecc0.npz")
    input = [
        t1,
        t2,
        t3,
        t4,
        t4 - t1,
        t3 - t2,
        jnp.array(rp_over_rs) * jnp.ones_like(t1),
        jnp.array(t0) * jnp.ones_like(t1),
        jnp.array(a_over_rs) * jnp.ones_like(t1),
        jnp.array(cosi * a_over_rs) * jnp.ones_like(t1),
        jnp.array(u1) * jnp.ones_like(t1),
        jnp.array(u2) * jnp.ones_like(t1),
        jnp.ones_like(t1),
        jitter * jnp.ones_like(t1),
    ]

    for param in posterior_sample:
        posterior_sample[param] = posterior_sample[param].reshape(
            (num_samples, *flux.shape[:-1])
        )
    t1_pred = posterior_sample["t0"] - posterior_sample["Ttot"] / 2
    t2_pred = posterior_sample["t0"] - posterior_sample["Tfull"] / 2
    t3_pred = posterior_sample["t0"] + posterior_sample["Tfull"] / 2
    t4_pred = posterior_sample["t0"] + posterior_sample["Ttot"] / 2
    pred_median = [
        jnp.median(t1_pred, axis=0),
        jnp.median(t2_pred, axis=0),
        jnp.median(t3_pred, axis=0),
        jnp.median(t4_pred, axis=0),
        jnp.median(posterior_sample["Ttot"], axis=0),
        jnp.median(posterior_sample["Tfull"], axis=0),
        jnp.median(posterior_sample["rp_over_rs"], axis=0),
        jnp.median(posterior_sample["t0"], axis=0),
        jnp.median(posterior_sample["a_over_rs"], axis=0),
        jnp.median(posterior_sample["cosi"] * posterior_sample["a_over_rs"], axis=0),
        jnp.median(posterior_sample["u1"], axis=0),
        jnp.median(posterior_sample["u2"], axis=0),
        jnp.median(posterior_sample["baseline"], axis=0),
        jnp.median(posterior_sample["jitter"], axis=0),
    ]
    pred_hpdi = [
        hpdi(t1_pred, 0.68),
        hpdi(t2_pred, 0.68),
        hpdi(t3_pred, 0.68),
        hpdi(t4_pred, 0.68),
        hpdi(posterior_sample["Ttot"], 0.68),
        hpdi(posterior_sample["Tfull"], 0.68),
        hpdi(posterior_sample["rp_over_rs"], 0.68),
        hpdi(posterior_sample["t0"], 0.68),
        hpdi(posterior_sample["a_over_rs"], 0.68),
        hpdi(posterior_sample["cosi"] * posterior_sample["a_over_rs"], 0.68),
        hpdi(posterior_sample["u1"], 0.68),
        hpdi(posterior_sample["u2"], 0.68),
        hpdi(posterior_sample["baseline"], 0.68),
        hpdi(posterior_sample["jitter"], 0.68),
    ]
    pred_yerr = jnp.array(
        [
            [median_i - hpdi_i[0], hpdi_i[1] - median_i]
            for median_i, hpdi_i in zip(pred_median, pred_hpdi)
        ]
    )
    xlabel = "Wavelength ($\mathrm{\mu m}$)"
    ylabel = [
        "$t_{\mathrm{I}}$",
        "$t_{\mathrm{II}}$",
        "$t_{\mathrm{III}}$",
        "$t_{\mathrm{IV}}$",
        "$T_{\mathrm{tot}}$",
        "$T_{\mathrm{full}}$",
        "$R_{\mathrm{p}}/R_{\mathrm{s}}$",
        "$t_0$",
        "$a/R_{\mathrm{s}}$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "baseline",
        "jitter",
    ]

    for i in range(len(ecc)):
        title_e = (
            rf" ($e\cos\omega$ ={ecc[i][0]*jnp.cos(omega[i][0]):.1f}, "
            + rf"$e\sin\omega$ ={ecc[i][0]*jnp.sin(omega[i][0]):.1f})"
        )
        filename_e = (
            f"_ecosw{ecc[i][0]*jnp.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*jnp.sin(omega[i][0]):.1f}_ecc0fix.png"
        )
        title = [ylabel_i + title_e for ylabel_i in ylabel]
        filename = [
            param_i + filename_e
            for param_i in [
                "t1",
                "t2",
                "t3",
                "t4",
                "Ttot",
                "Tfull",
                "rp",
                "t0",
                "a",
                "b",
                "u1",
                "u2",
                "baseline",
                "jitter",
            ]
        ]
        for j, title_j in enumerate(title):
            prediction_plot(
                wavelength,
                input[j][i],
                pred_median[j][i],
                pred_yerr[j][:, i],
                xlabel,
                ylabel[j],
                title_j,
                dir_output_ecc0,
                filename[j],
            )

    #################### eccentricity free ####################
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

    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(
        model,
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

    # SAMPLING
    posterior_sample = mcmc.get_samples()
    jnp.savez(dir_output + "posterior_sample", **posterior_sample)

    rng_key, rng_key_ = random.split(rng_key)
    pred = Predictive(model, posterior_sample)
    predictions = pred(
        rng_key_,
        flux_obs=None,
        time=time,
        num_lightcurve=int(jnp.prod(jnp.asarray(flux.shape[:-1]))),
    )
    jnp.savez(dir_output + "predictions", **predictions)
    # predictions = jnp.load(dir_output + "predictions_ecc0.npz")

    pred_light_curve = predictions["light_curve"].reshape((num_samples, *flux.shape))
    for i in range(len(ecc)):
        pred_median = jnp.median(pred_light_curve[:, i], axis=0)
        pred_hpdi = hpdi(pred_light_curve[:, i], 0.90)
        lightcurve_fit_plot(
            time,
            flux[i],
            pred_median,
            pred_hpdi,
            hpdi_range=0.90,
            title=f"Light Curve ($e\cos\omega$ ={ecc[i][0]*jnp.cos(omega[i][0]):.1f}, "
            + f"$e\sin\omega$ ={ecc[i][0]*jnp.sin(omega[i][0]):.1f})",
            dir_output=dir_output,
            filename=f"fit_lightcurve_ecosw{ecc[i][0]*jnp.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*jnp.sin(omega[i][0]):.1f}.png",
        )

    # posterior_sample = jnp.load(dir_output + "posterior_sample_ecc0.npz")
    input = [
        t1,
        t2,
        t3,
        t4,
        t4 - t1,
        t3 - t2,
        jnp.array(rp_over_rs) * jnp.ones_like(t1),
        jnp.array(t0) * jnp.ones_like(t1),
        jnp.array(a_over_rs) * jnp.ones_like(t1),
        jnp.array(ecc) * jnp.array(jnp.cos(omega)) * jnp.ones_like(t1),
        jnp.array(ecc) * jnp.array(jnp.sin(omega)) * jnp.ones_like(t1),
        jnp.array(cosi * a_over_rs) * jnp.ones_like(t1),
        jnp.array(u1) * jnp.ones_like(t1),
        jnp.array(u2) * jnp.ones_like(t1),
        jnp.ones_like(t1),
        jitter * jnp.ones_like(t1),
    ]

    for param in posterior_sample:
        posterior_sample[param] = posterior_sample[param].reshape(
            (num_samples, *flux.shape[:-1])
        )
    t1_pred, t2_pred, t3_pred, t4_pred = calc_contact_times(
        posterior_sample["rp_over_rs"],
        period,
        posterior_sample["a_over_rs"],
        posterior_sample["ecc"],
        posterior_sample["omega"],
        posterior_sample["cosi"],
        posterior_sample["t0"],
    )
    pred_median = [
        jnp.median(t1_pred, axis=0),
        jnp.median(t2_pred, axis=0),
        jnp.median(t3_pred, axis=0),
        jnp.median(t4_pred, axis=0),
        jnp.median(t4_pred - t1_pred, axis=0),
        jnp.median(t3_pred - t2_pred, axis=0),
        jnp.median(posterior_sample["rp_over_rs"], axis=0),
        jnp.median(posterior_sample["t0"], axis=0),
        jnp.median(posterior_sample["a_over_rs"], axis=0),
        jnp.median(posterior_sample["ecosw"], axis=0),
        jnp.median(posterior_sample["esinw"], axis=0),
        jnp.median(posterior_sample["b"], axis=0),
        jnp.median(posterior_sample["u1"], axis=0),
        jnp.median(posterior_sample["u2"], axis=0),
        jnp.median(posterior_sample["baseline"], axis=0),
        jnp.median(posterior_sample["jitter"], axis=0),
    ]
    pred_hpdi = [
        hpdi(t1_pred, 0.68),
        hpdi(t2_pred, 0.68),
        hpdi(t3_pred, 0.68),
        hpdi(t4_pred, 0.68),
        hpdi(t4_pred - t1_pred, 0.68),
        hpdi(t3_pred - t2_pred, 0.68),
        hpdi(posterior_sample["rp_over_rs"], 0.68),
        hpdi(posterior_sample["t0"], 0.68),
        hpdi(posterior_sample["a_over_rs"], 0.68),
        hpdi(posterior_sample["ecosw"], 0.68),
        hpdi(posterior_sample["esinw"], 0.68),
        hpdi(posterior_sample["b"], 0.68),
        hpdi(posterior_sample["u1"], 0.68),
        hpdi(posterior_sample["u2"], 0.68),
        hpdi(posterior_sample["baseline"], 0.68),
        hpdi(posterior_sample["jitter"], 0.68),
    ]
    pred_yerr = jnp.array(
        [
            [median_i - hpdi_i[0], hpdi_i[1] - median_i]
            for median_i, hpdi_i in zip(pred_median, pred_hpdi)
        ]
    )
    xlabel = "Wavelength ($\mathrm{\mu m}$)"
    ylabel = [
        "$t_{\mathrm{I}}$",
        "$t_{\mathrm{II}}$",
        "$t_{\mathrm{III}}$",
        "$t_{\mathrm{IV}}$",
        "$T_{\mathrm{tot}}$",
        "$T_{\mathrm{full}}$",
        "$R_{\mathrm{p}}/R_{\mathrm{s}}$",
        "$t_0$",
        "$a/R_{\mathrm{s}}$",
        "$e\cos\omega$",
        "$e\sin\omega$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "baseline",
        "jitter",
    ]

    for i in range(len(ecc)):
        title_e = (
            rf" ($e\cos\omega$ ={ecc[i][0]*jnp.cos(omega[i][0]):.1f}, "
            + rf"$e\sin\omega$ ={ecc[i][0]*jnp.sin(omega[i][0]):.1f})"
        )
        filename_e = (
            f"_ecosw{ecc[i][0]*jnp.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*jnp.sin(omega[i][0]):.1f}.png"
        )
        title = [ylabel_i + title_e for ylabel_i in ylabel]
        filename = [
            param_i + filename_e
            for param_i in [
                "t1",
                "t2",
                "t3",
                "t4",
                "Ttot",
                "Tfull",
                "rp",
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
        ]
        for j, title_j in enumerate(title):
            prediction_plot(
                wavelength,
                input[j][i],
                pred_median[j][i],
                pred_yerr[j][:, i],
                xlabel,
                ylabel[j],
                title_j,
                dir_output,
                filename[j],
            )
