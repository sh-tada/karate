from wasp39b_params import period_day
from karate.calc_contact_times import (
    calc_contact_times,
    calc_contact_times_circular,
    calc_contact_times_with_deltac,
)
from karate.convert_ctv import (
    contact_times_to_delta_c_ingress,
    contact_times_to_delta_c_egress,
    rotate_delta_c_egress,
    rotate_delta_c_ingress,
    dcx_to_rp_spectra,
)

import matplotlib.pyplot as plt
import corner
import numpy as np
from numpyro.diagnostics import hpdi

from jax import config
import jax
import jax.numpy as jnp

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def plot_2dlayout_grad(
    titles,
    ylabels,
    wavelength,
    values_list,
    truth_dc,
    dir_output="",
    filename="",
):
    print(filename)
    fontsize_pre = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 32
    fig = plt.figure(figsize=(10 * len(titles), 3 * len(ylabels[0])))
    for i, title in enumerate(titles):
        for j, ylabel in enumerate(ylabels[i]):
            ax = fig.add_subplot(len(ylabels[i]), len(titles), j * len(titles) + i + 1)
            if j == 0:
                ax.plot(
                    wavelength,
                    truth_dc[i],
                    color="black",
                    # marker=marker_list_truth[k][j][i],
                    # s=150,
                    # linestyle="None",
                    # color="0.0",
                    # ecolor="0.3",
                    # elinewidth=0.8,
                    # zorder=3,
                )
            ax.scatter(
                wavelength,
                values_list[j][i],
                color="black",
                # marker=marker_list_truth[k][j][i],
                s=50,
                # linestyle="None",
                # color="0.0",
                # ecolor="0.3",
                # elinewidth=0.8,
                # zorder=3,
            )
            if j != 0:
                ax.scatter(
                    wavelength,
                    values_list[j][i] * 2,
                    color="none",
                    # marker=marker_list_truth[k][j][i],
                    s=50,
                    # linestyle="None",
                    # color="0.0",
                    # ecolor="0.3",
                    # elinewidth=0.8,
                    # zorder=3,
                )
                ax.scatter(
                    wavelength,
                    -values_list[j][i] * 2,
                    color="none",
                    # marker=marker_list_truth[k][j][i],
                    s=50,
                    # linestyle="None",
                    # color="0.0",
                    # ecolor="0.3",
                    # elinewidth=0.8,
                    # zorder=3,
                )
            ax.set_ylabel(ylabel)
            # ax.legend()
            # ax.grid()
            ax.axhline(0, linestyle="--", color="black")
            if j == 0:
                ax.set_title(title)
            if j != len(ylabels[0]) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("Wavelength [nm]")

    # plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(dir_output + filename, bbox_inches="tight")
    # plt.show()
    plt.close()

    plt.rcParams["font.size"] = fontsize_pre


def convert_samples_rptn(posterior_samples, eccfree=False):
    if not eccfree:
        sample_t0 = posterior_samples["t0"]
        sample_Ttot = posterior_samples["Ttot"]
        sample_Tfull = posterior_samples["Tfull"]
        sample_depth = posterior_samples["depth"]
        sample_u1 = posterior_samples["u1"]
        sample_u2 = posterior_samples["u2"]
        sample_baseline = posterior_samples["baseline"]
        sample_jitter = posterior_samples["jitter"]

        theta_tot = 2 * np.pi * sample_Ttot / period
        theta_full = 2 * np.pi * sample_Tfull / period

        sample_rp_over_rs = np.sqrt(sample_depth)
        sample_a_over_rs = np.sqrt(
            (
                -((1 - np.sqrt(sample_depth)) ** 2) * np.cos(theta_tot / 2) ** 2
                + (1 + np.sqrt(sample_depth)) ** 2 * np.cos(theta_full / 2) ** 2
            )
            / np.sin((theta_tot + theta_full) / 2)
            / np.sin((theta_tot - theta_full) / 2)
        )
        sample_cosi = np.sqrt(
            (
                (1 - np.sqrt(sample_depth)) ** 2 * np.sin(theta_tot / 2) ** 2
                - (1 + np.sqrt(sample_depth)) ** 2 * np.sin(theta_full / 2) ** 2
            )
            / (
                -((1 - np.sqrt(sample_depth)) ** 2) * np.cos(theta_tot / 2) ** 2
                + (1 + np.sqrt(sample_depth)) ** 2 * np.cos(theta_full / 2) ** 2
            )
        )
        sample_b = sample_a_over_rs * sample_cosi
        sample_t1, sample_t2, sample_t3, sample_t4 = calc_contact_times_circular(
            sample_rp_over_rs, period, sample_a_over_rs, sample_cosi, sample_t0
        )
        orbit_dict = {
            "a_over_rs": sample_a_over_rs,
            "ecosw": np.zeros_like(sample_a_over_rs),
            "esinw": np.zeros_like(sample_a_over_rs),
            "cosi": sample_cosi,
            "t0": sample_t0,
        }
    elif eccfree:
        sample_rp_over_rs = posterior_samples["rp_over_rs"]
        sample_t0 = posterior_samples["t0"]
        sample_a_over_rs = posterior_samples["a_over_rs"]
        sample_ecosw = posterior_samples["ecosw"]
        sample_esinw = posterior_samples["esinw"]
        sample_b = posterior_samples["b"]
        sample_u1 = posterior_samples["u1"]
        sample_u2 = posterior_samples["u2"]
        sample_baseline = posterior_samples["baseline"]
        sample_jitter = posterior_samples["jitter"]

        sample_cosi = sample_b / sample_a_over_rs
        sample_ecc = np.sqrt(sample_ecosw**2 + sample_esinw**2)
        sample_omega = np.arctan2(sample_esinw, sample_ecosw)

        sample_t1, sample_t2, sample_t3, sample_t4 = calc_contact_times(
            sample_rp_over_rs,
            period,
            sample_a_over_rs,
            sample_ecc,
            sample_omega,
            sample_cosi,
            sample_t0,
        )
        orbit_dict = {
            "a_over_rs": sample_a_over_rs,
            "ecosw": sample_ecosw,
            "esinw": sample_esinw,
            "cosi": sample_cosi,
            "t0": sample_t0,
        }

    sample_dict = {
        "rp_over_rs": sample_rp_over_rs,
        "t1": sample_t1,
        "t2": sample_t2,
        "t3": sample_t3,
        "t4": sample_t4,
        "ti": (sample_t1 + sample_t2) / 2.0,
        "te": (sample_t3 + sample_t4) / 2.0,
        "taui": (sample_t2 - sample_t1),
        "taue": (sample_t4 - sample_t3),
    }
    return sample_dict, orbit_dict


def convert_samples_ctv(samples_dict_rptn, period, a_over_rs, ecc, omega, cosi, t0):
    dc_X_ingress, dc_Y_ingress = contact_times_to_delta_c_ingress(
        samples_dict_rptn["rp_over_rs"],
        samples_dict_rptn["t1"],
        samples_dict_rptn["t2"],
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        t0,
        rs_alpha=1,
    )
    dc_X_egress, dc_Y_egress = contact_times_to_delta_c_egress(
        samples_dict_rptn["rp_over_rs"],
        samples_dict_rptn["t3"],
        samples_dict_rptn["t4"],
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        t0,
        rs_alpha=1,
    )
    dc_xi_ingress, dc_yi_ingress = rotate_delta_c_ingress(
        dc_X_ingress, dc_Y_ingress, a_over_rs, ecc, omega, cosi
    )
    dc_xe_egress, dc_ye_egress = rotate_delta_c_egress(
        dc_X_egress, dc_Y_egress, a_over_rs, ecc, omega, cosi
    )
    rp_xip, rp_xin = dcx_to_rp_spectra(
        samples_dict_rptn["rp_over_rs"], dc_xi_ingress, rs_alpha=1
    )
    rp_xep, rp_xen = dcx_to_rp_spectra(
        samples_dict_rptn["rp_over_rs"], dc_xe_egress, rs_alpha=1
    )

    sample_dict = {
        "dc_over_rs_X_ingress": dc_X_ingress,
        "dc_over_rs_Y_ingress": dc_Y_ingress,
        "dc_over_rs_X_egress": dc_X_egress,
        "dc_over_rs_Y_egress": dc_Y_egress,
        "dc_over_rs_xi_ingress": dc_xi_ingress,
        "dc_over_rs_yi_ingress": dc_yi_ingress,
        "dc_over_rs_xe_egress": dc_xe_egress,
        "dc_over_rs_ye_egress": dc_ye_egress,
        "rp_xip": rp_xip,
        "rp_xin": rp_xin,
        "rp_xep": rp_xep,
        "rp_xen": rp_xen,
    }
    return sample_dict


def median_yerr(sample_dict):
    sample_median_dict = {}
    sample_err_dict = {}
    for param in sample_dict:
        sample_median = np.median(sample_dict[param], axis=0)
        sample_hpdi = hpdi(sample_dict[param], 0.68)
        sample_err = np.array(
            [sample_median - sample_hpdi[0], sample_hpdi[1] - sample_median]
        )
        sample_median_dict[param] = sample_median
        sample_err_dict[param] = sample_err
    return sample_median_dict, sample_err_dict


if __name__ == "__main__":
    default_fontsize = plt.rcParams["font.size"]
    fontsize = 28
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"  # LaTeXに近いスタイル
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    cmap = plt.get_cmap("viridis")

    wavelength = np.linspace(3.0, 5.0, 21)
    period = period_day * 24 * 60 * 60
    time = np.linspace(-150, 150, 301) * 65
    rp_over_rs = 0.150 + 0.005 * np.sin(wavelength * np.pi)
    t0 = 0
    a_over_rs = 11.4
    ecc = np.array([[0], [0.1], [0.1]])
    omega = np.array([[0], [0], [np.pi / 2]])
    cosi = 0.45 / 11.4
    u1 = 0.1
    u2 = 0.1
    jitter = 0.000
    dc_over_rs_X_ingress = 0.005 * np.cos(wavelength * 1.2 * np.pi + 0.5 * np.pi)
    dc_over_rs_Y_ingress = 0.005 * np.sin(wavelength * 1.6 * np.pi - 1.1 * np.pi)
    dc_over_rs_X_egress = 0.005 * np.cos(wavelength * 1.5 * np.pi + 0.1 * np.pi)
    dc_over_rs_Y_egress = 0.005 * np.sin(wavelength * 1.2 * np.pi + np.pi)

    ecosw = ecc * np.cos(omega)
    esinw = ecc * np.sin(omega)

    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )

    rp_over_rs_evening = 0.150 + 0.005 * np.sin(wavelength * np.pi)
    rp_over_rs_morning = 0.150 + 0.005 * np.sin(wavelength * np.pi * 1.6 + np.pi * 0.5)

    rp_over_rs_ingress = 0.150 * np.ones_like(wavelength)
    rp_over_rs_egress = 0.150 + 0.005 * np.sin(wavelength * np.pi)

    dc_over_rs_X_ingress = np.array(dc_over_rs_X_ingress)
    dc_over_rs_Y_ingress = np.array(dc_over_rs_Y_ingress)
    dc_over_rs_X_egress = np.array(dc_over_rs_X_egress)
    dc_over_rs_Y_egress = np.array(dc_over_rs_Y_egress)

    dc_over_rs_xi_ingress, dc_over_rs_yi_ingress = rotate_delta_c_ingress(
        dc_over_rs_X_ingress, dc_over_rs_Y_ingress, a_over_rs, ecc, omega, cosi
    )
    dc_over_rs_xe_egress, dc_over_rs_ye_egress = rotate_delta_c_egress(
        dc_over_rs_X_egress, dc_over_rs_Y_egress, a_over_rs, ecc, omega, cosi
    )

    rp_xip, rp_xin = dcx_to_rp_spectra(
        rp_over_rs, dc_over_rs_xi_ingress * np.ones_like(t1), rs_alpha=1
    )
    rp_xep, rp_xen = dcx_to_rp_spectra(
        rp_over_rs, dc_over_rs_xe_egress * np.ones_like(t1), rs_alpha=1
    )

    t1_dc, t2_dc, _, _ = calc_contact_times_with_deltac(
        rp_over_rs,
        dc_over_rs_X_ingress,
        dc_over_rs_Y_ingress,
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        t0,
    )
    _, _, t3_dc, t4_dc = calc_contact_times_with_deltac(
        rp_over_rs,
        dc_over_rs_X_egress,
        dc_over_rs_Y_egress,
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        t0,
    )

    num_samples = 1000
    dir_output = "mcmc_results/"

    # -----------------------------------------------------------------------
    # grad(dcxi, dcY, dcxe, dcY)

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    sample_median_dict, sample_err_dict = median_yerr(sample_dict)
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    def get_dc_xi(params_orbit, rp_over_rs, t1, t2):
        period, a_over_rs, ecosw, esinw, b, t0, rs_alpha = params_orbit
        ecc = jax.lax.cond(
            ecosw == 0 and esinw == 0,
            lambda _: 0.0,
            lambda _: jnp.sqrt(ecosw**2 + esinw**2),
            operand=None,
        )
        omega = jax.lax.cond(
            ecc == 0, lambda _: 0.0, lambda _: jnp.arctan2(esinw, ecosw), operand=None
        )
        cosi = b / a_over_rs
        dc_X_ingress, dc_Y_ingress = contact_times_to_delta_c_ingress(
            rp_over_rs,
            t1,
            t2,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
            rs_alpha,
        )
        dc_xi_ingress, dc_yi_ingress = rotate_delta_c_ingress(
            dc_X_ingress, dc_Y_ingress, a_over_rs, ecc, omega, cosi
        )
        return dc_xi_ingress

    def get_dc_Yi(params_orbit, rp_over_rs, t1, t2):
        period, a_over_rs, ecosw, esinw, b, t0, rs_alpha = params_orbit
        ecc = jax.lax.cond(
            ecosw == 0 and esinw == 0,
            lambda _: 0.0,
            lambda _: jnp.sqrt(ecosw**2 + esinw**2),
            operand=None,
        )
        omega = jax.lax.cond(
            ecc == 0, lambda _: 0.0, lambda _: jnp.arctan2(esinw, ecosw), operand=None
        )
        cosi = b / a_over_rs
        dc_X_ingress, dc_Y_ingress = contact_times_to_delta_c_ingress(
            rp_over_rs,
            t1,
            t2,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
            rs_alpha,
        )
        return dc_Y_ingress

    def get_dc_xe(params_orbit, rp_over_rs, t3, t4):
        period, a_over_rs, ecosw, esinw, b, t0, rs_alpha = params_orbit
        ecc = jax.lax.cond(
            ecosw == 0 and esinw == 0,
            lambda _: 0.0,
            lambda _: jnp.sqrt(ecosw**2 + esinw**2),
            operand=None,
        )
        omega = jax.lax.cond(
            ecc == 0, lambda _: 0.0, lambda _: jnp.arctan2(esinw, ecosw), operand=None
        )
        cosi = b / a_over_rs
        dc_X_egress, dc_Y_egress = contact_times_to_delta_c_egress(
            rp_over_rs,
            t3,
            t4,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
            rs_alpha,
        )
        dc_xe_egress, dc_ye_egress = rotate_delta_c_egress(
            dc_X_egress, dc_Y_egress, a_over_rs, ecc, omega, cosi
        )
        return dc_xe_egress

    def get_dc_Ye(params_orbit, rp_over_rs, t3, t4):
        period, a_over_rs, ecosw, esinw, b, t0, rs_alpha = params_orbit
        ecc = jax.lax.cond(
            ecosw == 0 and esinw == 0,
            lambda _: 0.0,
            lambda _: jnp.sqrt(ecosw**2 + esinw**2),
            operand=None,
        )
        omega = jax.lax.cond(
            ecc == 0, lambda _: 0.0, lambda _: jnp.arctan2(esinw, ecosw), operand=None
        )
        cosi = b / a_over_rs
        dc_X_egress, dc_Y_egress = contact_times_to_delta_c_egress(
            rp_over_rs,
            t3,
            t4,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
            rs_alpha,
        )
        return dc_Y_egress

    grad_get_dc_xi = jax.vmap(
        jax.grad(get_dc_xi),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    grad_get_dc_xe = jax.vmap(
        jax.grad(get_dc_xe),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    grad_get_dc_Yi = jax.vmap(
        jax.grad(get_dc_Yi),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    grad_get_dc_Ye = jax.vmap(
        jax.grad(get_dc_Ye),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    a_over_rs = np.mean(orbit_median_dict["a_over_rs"], axis=-1)[0]
    ecc = 10 ** (-3)
    omega = 0.0
    # omega = jnp.pi / 2
    cosi = np.mean(orbit_median_dict["cosi"], axis=-1)[0]
    t0 = np.mean(orbit_median_dict["t0"], axis=-1)[0]
    rs_alpha = 1.0

    rp_over_rs = sample_median_dict["rp_over_rs"][0]
    t1 = sample_median_dict["t1"][0]
    t2 = sample_median_dict["t2"][0]
    t3 = sample_median_dict["t3"][0]
    t4 = sample_median_dict["t4"][0]
    print(rp_over_rs, t1, t2, t3, t4)
    print(period, a_over_rs, ecc, omega, cosi * a_over_rs, t0, rs_alpha)

    dc_xi = get_dc_xi(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t1,
        t2,
    )
    grad_dc_xi = grad_get_dc_xi(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t1,
        t2,
    )
    dc_xe = get_dc_xe(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t3,
        t4,
    )
    grad_dc_xe = grad_get_dc_xe(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t3,
        t4,
    )

    dc_Yi = get_dc_Yi(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t1,
        t2,
    )
    grad_dc_Yi = grad_get_dc_Yi(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t1,
        t2,
    )
    dc_Ye = get_dc_Ye(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t3,
        t4,
    )
    grad_dc_Ye = grad_get_dc_Ye(
        [
            period,
            a_over_rs,
            ecc * jnp.cos(omega),
            ecc * jnp.sin(omega),
            cosi * a_over_rs,
            t0,
            rs_alpha,
        ],
        rp_over_rs,
        t3,
        t4,
    )

    titles = ["$f = \Delta c_{\mathrm{i,xi}}$", "$f = \Delta c_{\mathrm{e,xe}}$"]
    ylabel = [
        [
            "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial P$ [$R_{\mathrm{s}}/\mathrm{s}$]",
            r"$\partial f\, /\, \partial t_0$ [$R_{\mathrm{s}}/\mathrm{s}$]",
            r"$\partial f\, /\, \partial a$",
            r"$\partial f\, /\, \partial b$",
            r"$\partial f\, /\, \partial e\cos\omega$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial e\sin\omega$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial \alpha$ [$R_{\mathrm{s}}$]",
        ],
        [
            "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial P$ [$R_{\mathrm{s}}/\mathrm{s}$]",
            r"$\partial f\, /\, \partial t_0$ [$R_{\mathrm{s}}/\mathrm{s}$]",
            r"$\partial f\, /\, \partial a$",
            r"$\partial f\, /\, \partial b$",
            r"$\partial f\, /\, \partial e\cos\omega$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial e\sin\omega$ [$R_{\mathrm{s}}$]",
            r"$\partial f\, /\, \partial \alpha$ [$R_{\mathrm{s}}$]",
        ],
    ]

    truth_dc = [dc_over_rs_xi_ingress[0], dc_over_rs_xe_egress[0]]
    values_list = [
        np.array([dc_xi, dc_xe]),
        np.array([grad_dc_xi[0], grad_dc_xe[0]]),
        np.array([grad_dc_xi[5], grad_dc_xe[5]]),
        np.array([grad_dc_xi[1], grad_dc_xe[1]]),
        np.array([grad_dc_xi[4], grad_dc_xe[4]]),
        np.array([grad_dc_xi[2], grad_dc_xe[2]]),
        np.array([grad_dc_xi[3], grad_dc_xe[3]]),
        np.array([grad_dc_xi[6], grad_dc_xe[6]]),
    ]
    print(np.sum(values_list[0][0]), np.sum(values_list[0][1]))
    for values in values_list:
        print(0.001 / np.mean(values[0]), 0.001 / np.mean(values[1]))
    plot_2dlayout_grad(
        titles,
        ylabel,
        wavelength,
        values_list,
        truth_dc,
        dir_output=dir_output,
        filename="dc_grad_using_ecc0_from_dc_jitter0.png",
    )
