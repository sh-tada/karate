from wasp39b_params import period_day
from calc_light_curve import transit_compute_flux, transit_compute_flux_ecc0
from karate.calc_contact_times import calc_contact_times

import os
import sys
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random
from jax import config

import numpyro
from numpyro import distributions as dist
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi

# config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def model_ecc0(flux_obs, time, shape_params):
    period = period_day * 24 * 60 * 60
    t0 = numpyro.sample("t0", dist.Uniform(-5000, 5000).expand(shape_params))
    Ttot = numpyro.sample("Ttot", dist.Uniform(5000, 15000).expand(shape_params))
    Tfull = numpyro.sample("Tfull", dist.Uniform(jnp.ones_like(Ttot) * 1000, Ttot))

    theta_tot = 2 * jnp.pi * Ttot / period
    theta_full = 2 * jnp.pi * Tfull / period
    depth_max = (
        (jnp.sin(theta_tot / 2) - jnp.sin(theta_full / 2))
        / (jnp.sin(theta_tot / 2) + jnp.sin(theta_full / 2))
    ) ** 2
    depth = numpyro.sample("depth", dist.Uniform(jnp.zeros_like(depth_max), depth_max))

    u1 = numpyro.sample("u1", dist.Uniform(-3, 3).expand(shape_params))
    u2 = numpyro.sample("u2", dist.Uniform(-3, 3).expand(shape_params))
    baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01).expand(shape_params))
    jitter = numpyro.sample("jitter", dist.Uniform(0, 0.01).expand(shape_params))

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
    numpyro.sample(
        "light_curve",
        dist.Normal(
            flux * jnp.expand_dims(baseline, axis=-1),
            jnp.expand_dims(jitter, axis=-1) * jnp.ones_like(flux),
        ),
        obs=flux_obs,
    )


def model(flux_obs, time, shape_params):
    period = period_day * 24 * 60 * 60
    rp_over_rs = numpyro.sample("rp_over_rs", dist.Uniform(0, 0.5).expand(shape_params))
    t0 = numpyro.sample("t0", dist.Uniform(-5000, 5000).expand(shape_params))
    a_over_rs = numpyro.sample(
        "a_over_rs", dist.Uniform(-5000, 5000).expand(shape_params)
    )

    ecosw = numpyro.sample("ecosw", dist.Uniform(0, 1.0).expand(shape_params))
    esinw = numpyro.sample("esinw", dist.Uniform(0, 1.0).expand(shape_params))
    ecc = numpyro.deterministic("ecc", jnp.sqrt(ecosw**2 + esinw**2))
    omega = numpyro.deterministic(
        "omega", jnp.where(ecc > 0, jnp.arctan2(esinw / ecc, ecosw / ecc), 0.0)
    )

    b = numpyro.sample("b", dist.Uniform(0, 1.0).expand(shape_params))
    cosi = numpyro.deterministic("cosi", b / a_over_rs)

    u1 = numpyro.sample("u1", dist.Uniform(-3, 3).expand(shape_params))
    u2 = numpyro.sample("u2", dist.Uniform(-3, 3).expand(shape_params))
    baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01).expand(shape_params))
    jitter = numpyro.sample("jitter", dist.Uniform(0, 0.01).expand(shape_params))

    flux = transit_compute_flux(
        time, rp_over_rs, t0, period, a_over_rs, ecc, omega, cosi, u1, u2
    )
    numpyro.sample(
        "light_curve",
        dist.Normal(
            flux * jnp.expand_dims(baseline, axis=-1),
            jnp.expand_dims(jitter, axis=-1) * jnp.ones_like(flux),
        ),
        obs=flux_obs,
    )


def lightcurve_fit_plot(
    time,
    flux,
    pred_median,
    pred_hpdi,
    hpdi_range,
    title="",
    dir_output="",
    filename="light_curve_fit.png",
):
    """Make figures of lightcurve data and MCMC fit."""
    plt.figure(figsize=(12, 24))
    for i in range(len(flux)):
        plt.plot(
            time,
            flux[i] - 0.01 * i,
            marker=".",
            markersize=3,
            linestyle="None",
            # color="dodgerblue"
        )
        plt.plot(
            time,
            pred_median[i] - 0.01 * i,
            color="black",
            # lw=3,
            # zorder=5,
            # alpha=0.7,
        )
        plt.fill_between(
            time,
            pred_hpdi[0, i] - 0.01 * i,
            pred_hpdi[1, i] - 0.01 * i,
            alpha=0.2,
            interpolate=True,
            color="black",
            label=f"{int(hpdi_range*100)}% area",
        )
    # plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Relative Flux")
    plt.tight_layout()
    plt.savefig(dir_output + filename)
    plt.close()

    plt.figure(figsize=(12, 24))
    for i in range(len(flux)):
        plt.plot(
            time,
            flux[i] - pred_median[i] - 0.01 * i,
            marker=".",
            markersize=3,
            linestyle="None",
            # color="dodgerblue",
        )
        plt.fill_between(
            time,
            pred_hpdi[0, i] - pred_median[i] - 0.01 * i,
            pred_hpdi[1, i] - pred_median[i] - 0.01 * i,
            alpha=0.2,
            interpolate=True,
            color="black",
            label=f"{int(hpdi_range*100)}% area",
        )
    # plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Relative Flux")
    plt.tight_layout()
    plt.savefig(dir_output + filename.rsplit(".", 1)[0] + "_residual.png")
    plt.close()


def prediction_plot(
    x,
    input,
    prediction_median,
    prediction_hpdi_68,
    xlabel="",
    ylabel="",
    title="",
    dir_output="",
    filename="prediction.png",
):
    """Make a figure of MCMC fit."""
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.scatter(
        x,
        input,
        color="dodgerblue",
        marker="o",
        s=80,
        # linestyle="None",
        # color="0.0",
        # ecolor="0.3",
        # elinewidth=0.8,
        # zorder=3,
        label="Input",
    )
    ax1.errorbar(
        x,
        prediction_median,
        yerr=prediction_hpdi_68,
        color="black",
        marker="o",
        linestyle="None",
        # lw=3,
        # zorder=5,
        # alpha=0.7,
        label="Prediction",
    )
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.legend()

    ax2.errorbar(
        x,
        prediction_median - input,
        yerr=prediction_hpdi_68,
        color="black",
        marker="o",
        linestyle="None",
        # lw=3,
        # zorder=5,
        # alpha=0.7,
        label="Prediction - Input",
    )
    ax2.set_ylabel("Residual")
    ax2.set_xlabel(xlabel)
    ax2.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(dir_output + filename)
    plt.close()


if __name__ == "__main__":
    dir_output = "mcmc_results/"
    os.makedirs(dir_output, exist_ok=True)

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

    flux = transit_compute_flux(
        time, rp_over_rs, t0, period, a_over_rs, ecc, omega, cosi, u1, u2
    )
    error = 0.0005 * random.normal(rng_key_, shape=flux.shape)
    flux = flux + error
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

    num_warmup = 2000
    num_samples = 2000

    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(
        model_ecc0,
        forward_mode_differentiation=False,
        # max_tree_depth=13,
        # init_strategy=init_strategy,
        # target_accept_prob=0.9,
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        flux_obs=flux,
        time=time,
        shape_params=flux.shape[:-1],
    )

    mcmc.print_summary()
    with open(dir_output + "mcmc_summary_ecc0.txt", "w") as f:
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
    jnp.savez(dir_output + "posterior_sample_ecc0", **posterior_sample)

    rng_key, rng_key_ = random.split(rng_key)
    pred = Predictive(model_ecc0, posterior_sample)
    predictions = pred(
        rng_key_,
        flux_obs=None,
        time=time,
        shape_params=flux.shape[:-1],
    )
    jnp.savez(dir_output + "predictions_ecc0", **predictions)

    for i in range(len(ecc)):
        pred_median = jnp.median(predictions["light_curve"][:, i], axis=0)
        pred_hpdi = hpdi(predictions["light_curve"][:, i], 0.90)
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

    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )
    input = [
        jnp.array(rp_over_rs) * jnp.ones_like(t1),
        jnp.array(u1) * jnp.ones_like(t1),
        jnp.array(u2) * jnp.ones_like(t1),
        t1,
        t2,
        t3,
        t4,
    ]

    t1_pred = posterior_sample["t0"] - posterior_sample["Ttot"] / 2
    t2_pred = posterior_sample["t0"] - posterior_sample["Tfull"] / 2
    t3_pred = posterior_sample["t0"] + posterior_sample["Tfull"] / 2
    t4_pred = posterior_sample["t0"] + posterior_sample["Ttot"] / 2
    pred_median = [
        jnp.median(posterior_sample["rp_over_rs"], axis=0),
        jnp.median(posterior_sample["u1"], axis=0),
        jnp.median(posterior_sample["u2"], axis=0),
        jnp.median(t1_pred, axis=0),
        jnp.median(t2_pred, axis=0),
        jnp.median(t3_pred, axis=0),
        jnp.median(t4_pred, axis=0),
    ]
    pred_hpdi = [
        hpdi(posterior_sample["rp_over_rs"], 0.68),
        hpdi(posterior_sample["u1"], 0.68),
        hpdi(posterior_sample["u2"], 0.68),
        hpdi(t1_pred, 0.68),
        hpdi(t2_pred, 0.68),
        hpdi(t3_pred, 0.68),
        hpdi(t4_pred, 0.68),
    ]
    pred_yerr = jnp.array(
        [
            [median_i - hpdi_i[0], hpdi_i[1] - median_i]
            for median_i, hpdi_i in zip(pred_median, pred_hpdi)
        ]
    )
    xlabel = "Wavelength ($\mathrm{\mu m}$)"
    ylabel = [
        "$R_{\mathrm{p}}/R_{\mathrm{s}}$",
        "$u_1$",
        "$u_2$",
        "$t_{\mathrm{I}}$",
        "$t_{\mathrm{II}}$",
        "$t_{\mathrm{III}}$",
        "$t_{\mathrm{IV}}$",
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
        title = [
            ylabel_i + title_e
            for ylabel_i in [
                "$R_{\mathrm{p}}/R_{\mathrm{s}}$",
                "$u_1$",
                "$u_2$",
                "$t_{\mathrm{I}}$",
                "$t_{\mathrm{II}}$",
                "$t_{\mathrm{III}}$",
                "$t_{\mathrm{IV}}$",
            ]
        ]
        filename = [
            param_i + filename_e
            for param_i in ["rp", "u1", "u2", "t1", "t2", "t3", "t4"]
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
