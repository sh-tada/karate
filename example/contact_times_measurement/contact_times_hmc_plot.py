from wasp39b_params import period_day
from karate.calc_contact_times import (
    calc_contact_times,
    calc_contact_times_circular,
    calc_contact_times_with_deltac,
)

import matplotlib.pyplot as plt
import corner
import numpy as np
from numpyro.diagnostics import hpdi


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

    rp_over_rs = np.array(input_values_dict["rp_over_rs"])
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

    t1, t2, t3, t4 = calc_contact_times(
        rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0
    )
    truth_dict = {
        "rp_over_rs": rp_over_rs * np.ones_like(t1),
        "t0": t0 * np.ones_like(t1),
        "a_over_rs": a_over_rs * np.ones_like(t1),
        "ecosw": ecosw * cosi * np.ones_like(t1),
        "esinw": esinw * np.ones_like(t1),
        "b": a_over_rs * cosi * np.ones_like(t1),
        "u1": u1 * np.ones_like(t1),
        "u2": u2 * np.ones_like(t1),
        "baseline": 1.0 * np.ones_like(t1),
        "jitter": jitter * np.ones_like(t1),
        "depth": (rp_over_rs * np.ones_like(t1)) ** 2,
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
    label_dict = {
        "rp_over_rs": "$R_{\mathrm{p}}/R_{\mathrm{s}}$",
        "t0": "$t_0$",
        "a_over_rs": "$a/R_{\mathrm{s}}$",
        "ecosw": "$e\cos\omega$",
        "esinw": "$e\sin\omega$",
        "b": "$b$",
        "u1": "$u_1$",
        "u2": "$u_2$",
        "baseline": "baseline",
        "jitter": "jitter",
        "depth": "depth",
        "Ttot": "$T_{\mathrm{tot}}$",
        "Tfull": "$T_{\mathrm{full}}$",
        "t1": "$t_1$",
        "t2": "$t_2$",
        "t3": "$t_3$",
        "t4": "$t_4$",
        "ti": "$t_{\mathrm{i}}$",
        "te": "$t_{\mathrm{e}}$",
        "taui": r"$\tau_{\mathrm{i}}$",
        "taue": r"$\tau_{\mathrm{e}}$",
    }

    if deltac:
        dc_over_rs_x_ingress = np.array(input_values_dict["dc_over_rs_x_ingress"])
        dc_over_rs_y_ingress = np.array(input_values_dict["dc_over_rs_y_ingress"])
        dc_over_rs_x_egress = np.array(input_values_dict["dc_over_rs_x_egress"])
        dc_over_rs_y_egress = np.array(input_values_dict["dc_over_rs_y_egress"])

        t1_dc, t2_dc, _, _ = calc_contact_times_with_deltac(
            rp_over_rs,
            dc_over_rs_x_ingress,
            dc_over_rs_y_ingress,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
        )
        _, _, t3_dc, t4_dc = calc_contact_times_with_deltac(
            rp_over_rs,
            dc_over_rs_x_egress,
            dc_over_rs_y_egress,
            period,
            a_over_rs,
            ecc,
            omega,
            cosi,
            t0,
        )

        truth_dict["dc_over_rs_x_ingress"] = dc_over_rs_x_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_y_ingress"] = dc_over_rs_y_ingress * np.ones_like(t1)
        truth_dict["dc_over_rs_x_egress"] = dc_over_rs_x_egress * np.ones_like(t1)
        truth_dict["dc_over_rs_y_egress"] = dc_over_rs_y_egress * np.ones_like(t1)

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

        label_dict["dc_over_rs_x_ingress"] = "$\Delta c_{\mathrm{X}}$"
        label_dict["dc_over_rs_y_ingress"] = "$\Delta c_{\mathrm{Y}}$"
        label_dict["dc_over_rs_x_egress"] = "$\Delta c_{\mathrm{X}}$"
        label_dict["dc_over_rs_y_egress"] = "$\Delta c_{\mathrm{Y}}$"
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

        sample_t1, sample_t2, sample_t3, sample_t4 = calc_contact_times_circular(
            sample_rp_over_rs, period, sample_a_over_rs, sample_cosi, sample_t0
        )

        sample_dict = {
            "rp_over_rs": sample_rp_over_rs,
            "t0": sample_t0,
            "a_over_rs": sample_a_over_rs,
            "b": sample_a_over_rs * sample_cosi,
            "u1": sample_u1,
            "u2": sample_u2,
            "baseline": sample_baseline,
            "jitter": sample_jitter,
            "depth": sample_depth,
            "Ttot": sample_Ttot,
            "Tfull": sample_Tfull,
            "t1": sample_t1,
            "t2": sample_t2,
            "t3": sample_t3,
            "t4": sample_t4,
            "ti": (sample_t1 + sample_t2) / 2,
            "te": (sample_t3 + sample_t4) / 2,
            "taui": sample_t2 - sample_t1,
            "taue": sample_t4 - sample_t3,
        }

    elif fit_eccfree:
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

        sample_ecc = np.sqrt(sample_ecosw**2 + sample_esinw**2)
        sample_omega = np.where(
            ecc > 0, np.arctan2(sample_esinw / ecc, sample_ecosw / ecc), 0.0
        )

        sample_t1, sample_t2, sample_t3, sample_t4 = calc_contact_times_circular(
            sample_rp_over_rs,
            period,
            sample_a_over_rs,
            sample_ecc,
            sample_omega,
            sample_cosi,
            sample_t0,
        )

        sample_dict = {
            "rp_over_rs": sample_rp_over_rs,
            "t0": sample_t0,
            "a_over_rs": sample_a_over_rs,
            "ecosw": sample_ecosw,
            "esinw": sample_esinw,
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
        }

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
        "depth",
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
            title = label_dict[param] + title_e
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
            rf" (Input: $e\cos\omega$ ={ecc[i, 0]*np.cos(omega[i, 0]):.1f}, "
            + rf"$e\sin\omega$ ={ecc[i, 0]*np.sin(omega[i, 0]):.1f})"
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
                title = label_dict[param] + title_e
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
                title = label_dict[param] + title_e
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
    input_values_dict["dc_over_rs_x_ingress"] = 0.005 * np.cos(
        wavelength * 1.2 * np.pi + 0.5 * np.pi
    )
    input_values_dict["dc_over_rs_y_ingress"] = 0.005 * np.sin(
        wavelength * 1.5 * np.pi - 0.2 * np.pi
    )
    input_values_dict["dc_over_rs_x_egress"] = 0.005 * np.cos(
        wavelength * 1.5 * np.pi + 0.1 * np.pi
    )
    input_values_dict["dc_over_rs_y_egress"] = 0.005 * np.sin(
        wavelength * 1.2 * np.pi + np.pi
    )

    dir_output = "mcmc_results/dc_model_ecc0_jitter_00005/"
    num_samples = 1000

    plot_all(
        dir_output,
        input_values_dict,
        num_samples,
        dir_output + "flux.npy",
        dir_output + "posterior_sample_ecc0.npz",
        dir_output + "predictions_ecc0.npz",
        fit_eccfree=False,
        deltac=True,
    )
