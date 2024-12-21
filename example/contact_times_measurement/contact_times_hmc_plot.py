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

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


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
    truth,
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
        truth,
        color="dodgerblue",
        marker="o",
        s=80,
        # linestyle="None",
        # color="0.0",
        # ecolor="0.3",
        # elinewidth=0.8,
        # zorder=3,
        label="True values",
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
        label="Inferred values",
    )
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.legend()

    ax2.errorbar(
        x,
        prediction_median - truth,
        yerr=prediction_hpdi_68,
        color="black",
        marker="o",
        linestyle="None",
        # lw=3,
        # zorder=5,
        # alpha=0.7,
        label="Inferred - True",
    )
    ax2.set_ylabel("Residual")
    ax2.set_xlabel(xlabel)
    ax2.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(dir_output + filename)
    plt.close()


def prediction_plot_asym(
    x,
    truth,
    symmetrical_case,
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
        truth,
        color="dodgerblue",
        marker="o",
        s=80,
        # linestyle="None",
        # color="0.0",
        # ecolor="0.3",
        # elinewidth=0.8,
        # zorder=3,
        label="True values",
    )
    ax1.scatter(
        x,
        symmetrical_case,
        color="gray",
        marker="o",
        s=70,
        # linestyle="None",
        # color="0.0",
        # ecolor="0.3",
        # elinewidth=0.8,
        # zorder=3,
        label="Symmetrical case",
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
        label="Inferred values",
    )
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.legend()

    ax2.errorbar(
        x,
        prediction_median - truth,
        yerr=prediction_hpdi_68,
        color="black",
        marker="o",
        linestyle="None",
        # lw=3,
        # zorder=5,
        # alpha=0.7,
        label="Inferred - True",
    )
    ax2.set_ylabel("Residual")
    ax2.set_xlabel(xlabel)
    ax2.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(dir_output + filename)
    plt.close()


def plot_corner(
    sample_dict, var_names, labels="", truths="", dir_output="", filename=""
):
    fontsize_pre = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 10.0
    corner.corner(
        sample_dict,
        var_names=var_names,
        labels=labels,
        truths=truths,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        color="C0",
        # label_kwargs={"fontsize": 20},
        smooth=1.0,
    )
    plt.savefig(dir_output + filename)
    plt.close()
    plt.rcParams["font.size"] = fontsize_pre


def plot_all(
    dir_output,
    input_values_dict,
    num_samples,
    flux_file_path,
    sample_file_path,
    pred_file_path,
    fit_eccfree=False,
    deltac=False,
    catwoman=False,
    rp_change=False,
):
    flux = np.load(flux_file_path)
    posterior_samples = np.load(sample_file_path)
    posterior_samples = {key: posterior_samples[key] for key in posterior_samples}
    predictions = np.load(pred_file_path)
    predictions = {key: predictions[key] for key in predictions}

    pred_light_curve = predictions["light_curve"].reshape((num_samples, *flux.shape))
    for param in posterior_samples:
        posterior_samples[param] = posterior_samples[param].reshape(
            (num_samples, *flux.shape[:-1])
        )

    wavelength = np.array(input_values_dict["wavelength"])
    time = np.array(input_values_dict["time"])

    t0 = np.array(input_values_dict["t0"])
    period = np.array(input_values_dict["period"])
    a_over_rs = np.array(input_values_dict["a_over_rs"])
    ecc = np.array(input_values_dict["ecc"])
    omega = np.array(input_values_dict["omega"])
    cosi = np.array(input_values_dict["cosi"])
    u1 = np.array(input_values_dict["u1"])
    u2 = np.array(input_values_dict["u2"])
    jitter = np.array(input_values_dict["jitter"])

    ecosw = ecc * np.cos(omega)
    esinw = ecc * np.sin(omega)

    if not catwoman and not rp_change:
        rp_over_rs = np.array(input_values_dict["rp_over_rs"])
        t1, t2, t3, t4 = calc_contact_times(
            rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
        )
    elif catwoman:
        rp_over_rs_morning = np.array(input_values_dict["rp_over_rs_morning"])
        rp_over_rs_evening = np.array(input_values_dict["rp_over_rs_evening"])
        _, t2, _, t4 = calc_contact_times(
            rp_over_rs_morning, period, a_over_rs, ecc, omega, cosi, t0
        )
        t1, _, t3, _ = calc_contact_times(
            rp_over_rs_evening, period, a_over_rs, ecc, omega, cosi, t0
        )
    elif rp_change:
        rp_over_rs_ingress = np.array(input_values_dict["rp_over_rs_ingress"])
        rp_over_rs_egress = np.array(input_values_dict["rp_over_rs_egress"])
        t1, t2, _, _ = calc_contact_times(
            rp_over_rs_ingress, period, a_over_rs, ecc, omega, cosi, t0
        )
        _, _, t3, t4 = calc_contact_times(
            rp_over_rs_egress, period, a_over_rs, ecc, omega, cosi, t0
        )

    truth_dict = {
        "t0": t0 * np.ones_like(t1),
        "a_over_rs": a_over_rs * np.ones_like(t1),
        "ecosw": ecosw * cosi * np.ones_like(t1),
        "esinw": esinw * np.ones_like(t1),
        "b": a_over_rs * cosi * np.ones_like(t1),
        "u1": u1 * np.ones_like(t1),
        "u2": u2 * np.ones_like(t1),
        "baseline": 1.0 * np.ones_like(t1),
        "jitter": jitter * np.ones_like(t1),
        # "depth": (rp_over_rs * np.ones_like(t1)) ** 2,
        "Ttot": t4 - t1,
        "Tfull": t3 - t2,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
        "ti": (t1 + t2) / 2,
        "te": (t3 + t4) / 2,
        "taui": t2 - t1,
        "taue": t4 - t3,
    }
    if not catwoman and not rp_change:
        truth_dict["rp_over_rs"] = rp_over_rs * np.ones_like(t1)
    elif catwoman:
        truth_dict["rp_over_rs"] = np.sqrt(
            (rp_over_rs_morning**2 + rp_over_rs_evening**2) / 2
        ) * np.ones_like(t1)
        truth_dict["rp_over_rs_morning"] = rp_over_rs_morning * np.ones_like(t1)
        truth_dict["rp_over_rs_evening"] = rp_over_rs_evening * np.ones_like(t1)
    elif rp_change:
        truth_dict["rp_over_rs"] = (
            (rp_over_rs_ingress + rp_over_rs_egress) / 2.0 * np.ones_like(t1)
        )
        truth_dict["rp_over_rs_ingress"] = rp_over_rs_ingress * np.ones_like(t1)
        truth_dict["rp_over_rs_egress"] = rp_over_rs_egress * np.ones_like(t1)

    label_dict = {
        "rp_over_rs": "$R_{\mathrm{p}}$ [$R_{\mathrm{s}}$]",
        "rp_over_rs_morning": "$R_{\mathrm{p}}^{\mathrm{morning}}$ [$R_{\mathrm{s}}$]",
        "rp_over_rs_evening": "$R_{\mathrm{p}}^{\mathrm{evening}}$ [$R_{\mathrm{s}}$]",
        "rp_over_rs_ingress": "$R_{\mathrm{p}}^{\mathrm{ingress}}$ [$R_{\mathrm{s}}$]",
        "rp_over_rs_egress": "$R_{\mathrm{p}}^{\mathrm{egress}}$ [$R_{\mathrm{s}}$]",
        "t0": "$t_0$ [$\mathrm{s}$]",
        "a_over_rs": "$a$ [$R_{\mathrm{s}}$]",
        "ecosw": "$e\cos\omega$",
        "esinw": "$e\sin\omega$",
        "b": "$b$",
        "u1": "$u_1$",
        "u2": "$u_2$",
        "baseline": "baseline",
        "jitter": "jitter",
        "depth": "depth",
        "Ttot": "$T_{\mathrm{tot}} [\mathrm{s}]$",
        "Tfull": "$T_{\mathrm{full}} [\mathrm{s}]$",
        "t1": "$t_1 [\mathrm{s}]$",
        "t2": "$t_2 [\mathrm{s}]$",
        "t3": "$t_3 [\mathrm{s}]$",
        "t4": "$t_4 [\mathrm{s}]$",
        "ti": "$t_{\mathrm{i}} [\mathrm{s}]$",
        "te": "$t_{\mathrm{e}} [\mathrm{s}]$",
        "taui": r"$\tau_{\mathrm{i}} [\mathrm{s}]$",
        "taue": r"$\tau_{\mathrm{e}} [\mathrm{s}]$",
        "dc_over_rs_X_ingress": "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_Y_ingress": "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_X_egress": "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_Y_egress": "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_xi_ingress": "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_yi_ingress": "$\Delta c_{\mathrm{i,yi}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_xe_egress": "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]",
        "dc_over_rs_ye_egress": "$\Delta c_{\mathrm{e,ye}}$ [$R_{\mathrm{s}}$]",
        "rp_xip": "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        "rp_xin": "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        "rp_xep": "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        "rp_xen": "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    }

    if not deltac and not catwoman and not rp_change:
        truth_dict["dc_over_rs_X_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_X_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_xi_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_yi_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_xe_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_ye_egress"] = 0.0 * np.ones_like(t1)
        rp_xip, rp_xin = dcx_to_rp_spectra(
            rp_over_rs, 0.0 * np.ones_like(t1), rs_alpha=1
        )
        rp_xep, rp_xen = dcx_to_rp_spectra(
            rp_over_rs, 0.0 * np.ones_like(t1), rs_alpha=1
        )
        truth_dict["rp_xip"] = rp_xip
        truth_dict["rp_xin"] = rp_xin
        truth_dict["rp_xep"] = rp_xep
        truth_dict["rp_xen"] = rp_xen

    elif rp_change:
        truth_dict["dc_over_rs_X_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_X_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_xi_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_yi_ingress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_xe_egress"] = 0.0 * np.ones_like(t1)
        truth_dict["dc_over_rs_ye_egress"] = 0.0 * np.ones_like(t1)
        rp_xip, rp_xin = dcx_to_rp_spectra(
            rp_over_rs_ingress, 0.0 * np.ones_like(t1), rs_alpha=1
        )
        rp_xep, rp_xen = dcx_to_rp_spectra(
            rp_over_rs_egress, 0.0 * np.ones_like(t1), rs_alpha=1
        )
        truth_dict["rp_xip"] = rp_xip
        truth_dict["rp_xin"] = rp_xin
        truth_dict["rp_xep"] = rp_xep
        truth_dict["rp_xen"] = rp_xen

    elif deltac:
        dc_over_rs_X_ingress = np.array(input_values_dict["dc_over_rs_X_ingress"])
        dc_over_rs_Y_ingress = np.array(input_values_dict["dc_over_rs_Y_ingress"])
        dc_over_rs_X_egress = np.array(input_values_dict["dc_over_rs_X_egress"])
        dc_over_rs_Y_egress = np.array(input_values_dict["dc_over_rs_Y_egress"])

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
        truth_dict["rp_xip"] = rp_xip
        truth_dict["rp_xin"] = rp_xin
        truth_dict["rp_xep"] = rp_xep
        truth_dict["rp_xen"] = rp_xen

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

        truth_dict["dc_over_rs_X_ingress"] = dc_over_rs_X_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_ingress"] = dc_over_rs_Y_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_X_egress"] = dc_over_rs_X_egress * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_egress"] = dc_over_rs_Y_egress * np.ones_like(t1)

        truth_dict["dc_over_rs_xi_ingress"] = dc_over_rs_xi_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_yi_ingress"] = dc_over_rs_yi_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_xe_egress"] = dc_over_rs_xe_egress * np.ones_like(t1)
        truth_dict["dc_over_rs_ye_egress"] = dc_over_rs_ye_egress * np.ones_like(t1)

        truth_dict["t1_with_deltac"] = t1_dc
        truth_dict["t2_with_deltac"] = t2_dc
        truth_dict["t3_with_deltac"] = t3_dc
        truth_dict["t4_with_deltac"] = t4_dc

        truth_dict["ti_with_deltac"] = (t1_dc + t2_dc) / 2.0
        truth_dict["te_with_deltac"] = (t3_dc + t4_dc) / 2.0
        truth_dict["taui_with_deltac"] = t2_dc - t1_dc
        truth_dict["taue_with_deltac"] = t4_dc - t3_dc
        truth_dict["Ttot_with_deltac"] = t4_dc - t1_dc
        truth_dict["Tfull_with_deltac"] = t3_dc - t2_dc

        label_dict["t1_with_deltac"] = "$t_1$ with $\Delta c$"
        label_dict["t2_with_deltac"] = "$t_2$ with $\Delta c$"
        label_dict["t3_with_deltac"] = "$t_3$ with $\Delta c$"
        label_dict["t4_with_deltac"] = "$t_4$ with $\Delta c$"
        label_dict["ti_with_deltac"] = "$t_{\mathrm{i}}$ with $\Delta c$"
        label_dict["te_with_deltac"] = "$t_{\mathrm{e}}$ with $\Delta c$"
        label_dict["taui_with_deltac"] = r"$\tau_{\mathrm{i}}$ with $\Delta c$"
        label_dict["taue_with_deltac"] = r"$\tau_{\mathrm{e}}$ with $\Delta c$"
        label_dict["Ttot_with_deltac"] = "$T_{\mathrm{tot}}$ with $\Delta c$"
        label_dict["Tfull_with_deltac"] = "$T_{\mathrm{full}}$ with $\Delta c$"

    elif catwoman:
        dc_over_rs_X_ingress = (rp_over_rs_morning - rp_over_rs_evening) / 2.0
        dc_over_rs_Y_ingress = 0
        dc_over_rs_X_egress = (rp_over_rs_morning - rp_over_rs_evening) / 2.0
        dc_over_rs_Y_egress = 0

        dc_over_rs_xi_ingress, dc_over_rs_yi_ingress = rotate_delta_c_ingress(
            dc_over_rs_X_ingress, dc_over_rs_Y_ingress, a_over_rs, ecc, omega, cosi
        )
        dc_over_rs_xe_egress, dc_over_rs_ye_egress = rotate_delta_c_egress(
            dc_over_rs_X_egress, dc_over_rs_Y_egress, a_over_rs, ecc, omega, cosi
        )

        truth_dict["dc_over_rs_X_ingress"] = dc_over_rs_X_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_ingress"] = dc_over_rs_Y_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_X_egress"] = dc_over_rs_X_egress * np.ones_like(t1)
        truth_dict["dc_over_rs_Y_egress"] = dc_over_rs_Y_egress * np.ones_like(t1)

        truth_dict["dc_over_rs_xi_ingress"] = dc_over_rs_xi_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_yi_ingress"] = dc_over_rs_yi_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_xe_egress"] = dc_over_rs_xe_egress * np.ones_like(t1)
        truth_dict["dc_over_rs_ye_egress"] = dc_over_rs_ye_egress * np.ones_like(t1)

        truth_dict["rp_xip"] = rp_over_rs_morning * np.ones_like(t1)
        truth_dict["rp_xin"] = rp_over_rs_evening * np.ones_like(t1)
        truth_dict["rp_xep"] = rp_over_rs_morning * np.ones_like(t1)
        truth_dict["rp_xen"] = rp_over_rs_evening * np.ones_like(t1)

    # light curve
    for i in range(len(ecc)):
        pred_median = np.median(pred_light_curve[:, i], axis=0)
        pred_hpdi = hpdi(pred_light_curve[:, i], 0.90)
        lightcurve_fit_plot(
            time,
            flux[i],
            pred_median,
            pred_hpdi,
            hpdi_range=0.90,
            title=f"Light Curve ($e\cos\omega$ ={ecc[i][0]*np.cos(omega[i][0]):.1f}, "
            + f"$e\sin\omega$ ={ecc[i][0]*np.sin(omega[i][0]):.1f})",
            dir_output=dir_output,
            filename=f"fit_lightcurve_ecosw{ecc[i][0]*np.cos(omega[i][0]):.1f}"
            + f"_esinw{ecc[i][0]*np.sin(omega[i][0]):.1f}.png",
        )

    # parameters
    if not fit_eccfree:
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

    else:
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

    dc_X_ingress, dc_Y_ingress = contact_times_to_delta_c_ingress(
        sample_rp_over_rs,
        sample_t1,
        sample_t2,
        period,
        a_over_rs,
        ecc,
        omega,
        cosi,
        t0,
        rs_alpha=1,
    )
    dc_X_egress, dc_Y_egress = contact_times_to_delta_c_egress(
        sample_rp_over_rs,
        sample_t3,
        sample_t4,
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
    rp_xip, rp_xin = dcx_to_rp_spectra(sample_rp_over_rs, dc_xi_ingress, rs_alpha=1)
    rp_xep, rp_xen = dcx_to_rp_spectra(sample_rp_over_rs, dc_xe_egress, rs_alpha=1)

    sample_dict = {
        "rp_over_rs": sample_rp_over_rs,
        "t0": sample_t0,
        "a_over_rs": sample_a_over_rs,
        "b": sample_b,
        "u1": sample_u1,
        "u2": sample_u2,
        "baseline": sample_baseline,
        "jitter": sample_jitter,
        "depth": sample_rp_over_rs**2,
        "Ttot": sample_t4 - sample_t1,
        "Tfull": sample_t3 - sample_t2,
        "t1": sample_t1,
        "t2": sample_t2,
        "t3": sample_t3,
        "t4": sample_t4,
        "ti": (sample_t1 + sample_t2) / 2,
        "te": (sample_t3 + sample_t4) / 2,
        "taui": sample_t2 - sample_t1,
        "taue": sample_t4 - sample_t3,
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
    if fit_eccfree:
        sample_dict["ecosw"] = sample_ecosw
        sample_dict["esinw"] = sample_esinw

    # plot parameters
    xlabel = "Wavelength ($\mathrm{\mu m}$)"
    params = [
        "rp_over_rs",
        "t0",
        "a_over_rs",
        "b",
        "u1",
        "u2",
        "baseline",
        "jitter",
        "dc_over_rs_X_ingress",
        "dc_over_rs_Y_ingress",
        "dc_over_rs_X_egress",
        "dc_over_rs_Y_egress",
        "dc_over_rs_xi_ingress",
        "dc_over_rs_yi_ingress",
        "dc_over_rs_xe_egress",
        "dc_over_rs_ye_egress",
        "rp_xip",
        "rp_xin",
        "rp_xep",
        "rp_xen",
    ]
    if fit_eccfree:
        params.append("ecosw")
        params.append("esinw")
    for i in range(len(ecc)):
        title_e = (
            rf" (Input: $e\cos\omega$ ={ecc[i, 0]*np.cos(omega[i, 0]):.1f}, "
            + rf"$e\sin\omega$ ={ecc[i, 0]*np.sin(omega[i, 0]):.1f})"
        )
        filename_e = (
            f"_ecosw{ecc[i, 0]*np.cos(omega[i, 0]):.1f}"
            + f"_esinw{ecc[i, 0]*np.sin(omega[i, 0]):.1f}.png"
        )

        for param in params:
            sample_median = np.median(sample_dict[param][:, i], axis=0)
            sample_hpdi = hpdi(sample_dict[param][:, i], 0.68)
            sample_yerr = np.array(
                [sample_median - sample_hpdi[0], sample_hpdi[1] - sample_median]
            )
            filename = param + filename_e
            title = label_dict[param].rsplit(" [", -1)[0] + title_e
            prediction_plot(
                wavelength,
                truth_dict[param][i],
                sample_median,
                sample_yerr,
                xlabel,
                label_dict[param],
                title,
                dir_output,
                filename,
            )

    params = [
        "Ttot",
        "Tfull",
        "t1",
        "t2",
        "t3",
        "t4",
        "ti",
        "te",
        "taui",
        "taue",
    ]
    for i in range(len(ecc)):
        title_e = (
            f" (Input: $e\cos\omega$ ={ecc[i, 0]*np.cos(omega[i, 0]):.1f}, "
            + f"$e\sin\omega$ ={ecc[i, 0]*np.sin(omega[i, 0]):.1f})"
        )
        filename_e = (
            f"_ecosw{ecc[i, 0]*np.cos(omega[i, 0]):.1f}"
            + f"_esinw{ecc[i, 0]*np.sin(omega[i, 0]):.1f}.png"
        )
        if not deltac:
            for param in params:
                sample_median = np.median(sample_dict[param][:, i], axis=0)
                sample_hpdi = hpdi(sample_dict[param][:, i], 0.68)
                sample_yerr = np.array(
                    [sample_median - sample_hpdi[0], sample_hpdi[1] - sample_median]
                )
                filename = param + filename_e
                title = label_dict[param].rsplit(" [", -1)[0] + title_e
                prediction_plot(
                    wavelength,
                    truth_dict[param][i],
                    sample_median,
                    sample_yerr,
                    xlabel,
                    label_dict[param],
                    title,
                    dir_output,
                    filename,
                )
        elif deltac:
            for param in params:
                sample_median = np.median(sample_dict[param][:, i], axis=0)
                sample_hpdi = hpdi(sample_dict[param][:, i], 0.68)
                sample_yerr = np.array(
                    [sample_median - sample_hpdi[0], sample_hpdi[1] - sample_median]
                )
                filename = param + filename_e
                title = label_dict[param].rsplit(" [", -1)[0] + title_e
                prediction_plot_asym(
                    wavelength,
                    truth_dict[param + "_with_deltac"][i],
                    truth_dict[param][i],
                    sample_median,
                    sample_yerr,
                    xlabel,
                    label_dict[param],
                    title,
                    dir_output,
                    filename,
                )

    # corner
    if not fit_eccfree:
        var_names = [
            "rp_over_rs",
            "ti",
            "te",
            "taui",
            "taue",
            "u1",
            "u2",
            "baseline",
            "jitter",
        ]
    if fit_eccfree:
        var_names = [
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
    for i in range(len(ecc)):
        sample_dict_plot = {}
        for j in range(len(wavelength)):
            if j % 10 != 0:
                continue
            for var in var_names:
                sample_dict_plot[var] = sample_dict[var][:, i, j]
            labels = [label_dict[var] for var in var_names]
            truths = [truth_dict[var][i, j] for var in var_names]
            if deltac:
                for k, var in enumerate(var_names):
                    if var in ["ti", "te", "taui", "taue"]:
                        truths[k] = truth_dict[var + "_with_deltac"][i, j]
            filename_e = (
                f"_ecosw{ecc[i, 0]*np.cos(omega[i, 0]):.1f}"
                + f"_esinw{ecc[i, 0]*np.sin(omega[i, 0]):.1f}"
            )
            filename = f"corner_{wavelength[j]:.1f}um" + filename_e + ".png"
            plot_corner(
                sample_dict_plot, var_names, labels, truths, dir_output, filename
            )


if __name__ == "__main__":
    default_fontsize = plt.rcParams["font.size"]
    fontsize = 28
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"  # LaTeXに近いスタイル
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    wavelength = np.linspace(3.0, 5.0, 21)
    input_values_dict = {}
    input_values_dict["wavelength"] = wavelength
    input_values_dict["time"] = np.linspace(-150, 150, 301) * 65
    input_values_dict["rp_over_rs"] = 0.150 + 0.005 * np.sin(wavelength * np.pi)
    input_values_dict["t0"] = 0
    input_values_dict["period"] = period_day * 24 * 60 * 60
    input_values_dict["a_over_rs"] = 11.4
    input_values_dict["ecc"] = [[0], [0.1], [0.1]]
    input_values_dict["omega"] = [[0], [0], [np.pi / 2]]
    input_values_dict["cosi"] = 0.45 / 11.4
    input_values_dict["u1"] = 0.1
    input_values_dict["u2"] = 0.1
    input_values_dict["jitter"] = 0.000
    input_values_dict["dc_over_rs_X_ingress"] = 0.005 * np.cos(
        wavelength * 1.2 * np.pi + 0.5 * np.pi
    )
    input_values_dict["dc_over_rs_Y_ingress"] = 0.005 * np.sin(
        wavelength * 1.5 * np.pi - 0.2 * np.pi
    )
    input_values_dict["dc_over_rs_X_egress"] = 0.005 * np.cos(
        wavelength * 1.5 * np.pi + 0.1 * np.pi
    )
    input_values_dict["dc_over_rs_Y_egress"] = 0.005 * np.sin(
        wavelength * 1.2 * np.pi + np.pi
    )

    num_samples = 1000

    dir_output = "mcmc_results/dc_model_ecc0_jitter_0/"
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

    dir_output = "mcmc_results/dc_model_eccfree_jitter_0/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=True,
        deltac=True,
    )

    dir_output = "mcmc_results/dc_model_ecc0_jitter_00005/"
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

    dir_output = "mcmc_results/dc_model_eccfree_jitter_00005/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=True,
        deltac=True,
    )

    dir_output = "mcmc_results/model_ecc0_jitter_0/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=False,
        deltac=False,
    )

    dir_output = "mcmc_results/model_eccfree_jitter_0/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=True,
        deltac=False,
    )

    dir_output = "mcmc_results/model_ecc0_jitter_00005/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=False,
        deltac=False,
    )

    dir_output = "mcmc_results/model_eccfree_jitter_00005/"
    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample.npz",
        dir_output + "predictions.npz",
        fit_eccfree=True,
        deltac=False,
    )
