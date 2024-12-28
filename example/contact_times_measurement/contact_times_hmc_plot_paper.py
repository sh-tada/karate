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


def plot_vertical(
    ylabels,
    wavelength,
    truth_list,
    median_list,
    yerr_list,
    label_list_truth,
    color_list_truth,
    marker_list_truth,
    label_list_inferred,
    color_list_inferred,
    marker_list_inferred,
    dir_output="",
    filename="",
):
    fontsize_pre = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 32

    fig = plt.figure(figsize=(18, 3 * len(ylabels)))
    for i, ylabel in enumerate(ylabels):
        ax = fig.add_subplot(len(ylabels), 1, i + 1)
        for j, truth in enumerate(truth_list[i]):
            ax.scatter(
                wavelength,
                truth,
                color=color_list_truth[i][j],
                marker=marker_list_truth[i][j],
                s=100,
                # linestyle="None",
                # color="0.0",
                # ecolor="0.3",
                # elinewidth=0.8,
                # zorder=3,
                label=label_list_truth[i][j],
            )
        for j, median in enumerate(median_list[i]):
            ax.errorbar(
                wavelength,
                median,
                yerr=yerr_list[i][:, j, :],
                color=color_list_inferred[i][j],
                marker=marker_list_inferred[i][j],
                linestyle="None",
                # lw=3,
                # zorder=5,
                # alpha=0.7,
                label=label_list_inferred[i][j],
            )
        ax.set_ylabel(ylabel)
        # ax.legend()
        ax.grid()
        if i != len(ylabels) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("Wavelength [nm]")

    axLine, axLabel = ax.get_legend_handles_labels()
    # fig.legend(lines[4:], labels[4:], bbox_to_anchor=(1.01, 0.98), loc="upper left")

    # plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    fig.legend(axLine, axLabel, bbox_to_anchor=(0.2, 0.01), loc="upper left", ncol=3)
    plt.savefig(dir_output + filename, bbox_inches="tight")
    # plt.show()
    plt.close()

    plt.rcParams["font.size"] = fontsize_pre


def plot_2dlayout(
    titles,
    ylabels,
    wavelength,
    truth_list,
    median_list,
    yerr_list,
    label_list_truth,
    color_list_truth,
    marker_list_truth,
    label_list_inferred,
    color_list_inferred,
    marker_list_inferred,
    len_true_values=1,
    len_inferred_values=1,
    dir_output="",
    filename="",
    bbox_to_anchor=[0.2, 0.3],
    ncol=3,
):
    print(filename)
    fontsize_pre = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 32
    if len_true_values == 1:
        truth_list = [truth_list]
        label_list_truth = [label_list_truth]
        color_list_truth = [color_list_truth]
        marker_list_truth = [marker_list_truth]
    if len_inferred_values == 1:
        median_list = [median_list]
        yerr_list = [yerr_list]
        label_list_inferred = [label_list_inferred]
        color_list_inferred = [color_list_inferred]
        marker_list_inferred = [marker_list_inferred]

    if len(titles) == 3:
        fig = plt.figure(figsize=(8 * len(titles), 3 * len(ylabels)))
    if len(titles) == 1:
        fig = plt.figure(figsize=(18 * len(titles), 4 * len(ylabels)))
    else:
        fig = plt.figure(figsize=(10 * len(titles), 3 * len(ylabels)))
    for i, title in enumerate(titles):
        for j, ylabel in enumerate(ylabels):
            ax = fig.add_subplot(len(ylabels), len(titles), j * len(titles) + i + 1)
            for k in range(len_true_values):
                ax.plot(
                    wavelength,
                    truth_list[k][j][i],
                    color=color_list_truth[k][j][i],
                    # marker=marker_list_truth[k][j][i],
                    # s=150,
                    # linestyle="None",
                    # color="0.0",
                    # ecolor="0.3",
                    # elinewidth=0.8,
                    # zorder=3,
                    label=label_list_truth[k][j][i],
                )
            for k in range(len_inferred_values):
                if marker_list_inferred[k][j][i] != "s":
                    ax.errorbar(
                        wavelength,
                        median_list[k][j][i],
                        yerr=yerr_list[k][j][:, i, :],
                        ecolor=color_list_inferred[k][j][i],
                        markerfacecolor=color_list_inferred[k][j][i],
                        markeredgecolor=color_list_inferred[k][j][i],
                        marker=marker_list_inferred[k][j][i],
                        markersize=8,
                        linestyle="None",
                        # lw=3,
                        # zorder=5,
                        # alpha=0.7,
                        label=label_list_inferred[k][j][i],
                    )
                else:
                    ax.errorbar(
                        wavelength,
                        median_list[k][j][i],
                        yerr=yerr_list[k][j][:, i, :],
                        ecolor="grey",
                        markerfacecolor=color_list_inferred[k][j][i],
                        markeredgecolor="grey",
                        marker=marker_list_inferred[k][j][i],
                        markersize=8,
                        linestyle="None",
                        # lw=3,
                        # zorder=5,
                        # alpha=0.7,
                        label=label_list_inferred[k][j][i],
                    )
            ax.set_ylabel(ylabel)
            # ax.legend()
            # ax.grid()
            # ax.axhline(0)
            if j == 0:
                ax.set_title(title)
            if j != len(ylabels) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j == len(ylabels) - 1 and i != len(titles) - 1:
                axLine, axLabel = ax.get_legend_handles_labels()
            if j == len(ylabels) - 1 and len(titles) == 1:
                axLine, axLabel = ax.get_legend_handles_labels()
        if j == len(ylabels) - 1 and title == "With noise":
            axLine_e, axLabel_e = ax.get_legend_handles_labels()
            axLine = axLine + axLine_e[-2:]
            axLabel = axLabel + axLabel_e[-2:]
        ax.set_xlabel("Wavelength [nm]")

    # fig.legend(lines[4:], labels[4:], bbox_to_anchor=(1.01, 0.98), loc="upper left")

    num_items = len(axLine)
    rows = (num_items - 1) // ncol + 1
    ordered_indices = [
        c + r * ncol
        for c in range(ncol)
        for r in range(rows)
        if c + r * ncol < num_items
    ]
    print(axLabel)
    axLine = [axLine[i] for i in ordered_indices]
    axLabel = [axLabel[i] for i in ordered_indices]
    print(axLabel)

    # plt.tick_params(labelsize=16)
    plt.tight_layout()
    if len(titles) == 3:
        plt.subplots_adjust(hspace=0.3)
    else:
        plt.subplots_adjust(hspace=0.1)
    fig.legend(
        axLine,
        axLabel,
        bbox_to_anchor=(bbox_to_anchor[0], 0.01),
        loc="upper left",
        ncol=ncol,
    )
    plt.savefig(dir_output + filename, bbox_inches="tight")
    # plt.show()
    plt.close()

    if len(titles) == 3:
        fig = plt.figure(figsize=(8 * len(titles), 3 * len(ylabels)))
    else:
        fig = plt.figure(figsize=(10 * len(titles), 3 * len(ylabels)))
    for i, title in enumerate(titles):
        for j, ylabel in enumerate(ylabels):
            ax = fig.add_subplot(len(ylabels), len(titles), j * len(titles) + i + 1)
            for k in range(len_inferred_values):
                if marker_list_inferred[k][j][i] != "s":
                    ax.errorbar(
                        wavelength,
                        median_list[k][j][i] - truth_list[0][j][i],
                        yerr=yerr_list[k][j][:, i, :],
                        color=color_list_inferred[k][j][i],
                        marker=marker_list_inferred[k][j][i],
                        markersize=8,
                        linestyle="None",
                        # lw=3,
                        # zorder=5,
                        # alpha=0.7,
                        label="Inferred - True"
                        + label_list_inferred[k][j][i].rsplit("values", 1)[-1],
                    )
                else:
                    ax.errorbar(
                        wavelength,
                        median_list[k][j][i] - truth_list[0][j][i],
                        yerr=yerr_list[k][j][:, i, :],
                        color=color_list_inferred[k][j][i],
                        markeredgecolor="none",
                        marker=marker_list_inferred[k][j][i],
                        markersize=8,
                        linestyle="None",
                        # lw=3,
                        # zorder=5,
                        # alpha=0.7,
                        label="Inferred - True"
                        + label_list_inferred[k][j][i].rsplit("values", 1)[-1],
                    )
            ax.set_ylabel(ylabel)
            # ax.legend()
            # ax.grid()
            ax.axhline(0, color="black")
            if j == 0:
                ax.set_title(title)
            if j != len(ylabels) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("Wavelength [nm]")

    axLine, axLabel = ax.get_legend_handles_labels()
    # fig.legend(lines[4:], labels[4:], bbox_to_anchor=(1.01, 0.98), loc="upper left")

    # plt.tick_params(labelsize=16)
    plt.tight_layout()
    if len(titles) == 3:
        plt.subplots_adjust(hspace=0.3)
    else:
        plt.subplots_adjust(hspace=0.1)
    fig.legend(
        axLine,
        axLabel,
        bbox_to_anchor=(bbox_to_anchor[1], 0.01),
        loc="upper left",
        ncol=3,
    )
    plt.savefig(
        dir_output + filename.rsplit(".", 1)[0] + "_residual.png", bbox_inches="tight"
    )
    # plt.show()
    plt.close()

    plt.rcParams["font.size"] = fontsize_pre


def plot_rp4(
    wavelength,
    truth_list,
    median_list,
    yerr_list,
    label_list_truth,
    color_list_truth,
    marker_list_truth,
    label_list_inferred,
    color_list_inferred,
    marker_list_inferred,
    len_true_values=1,
    len_inferred_values=1,
    dir_output="",
    filename="",
    bbox_to_anchor=0.2,
    ncol=3,
    ls="-",
    ylim=[0.13, 0.167],
):
    print(filename)
    fontsize_pre = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 32
    if len_true_values == 1:
        truth_list = [truth_list]
        label_list_truth = [label_list_truth]
        color_list_truth = [color_list_truth]
        marker_list_truth = [marker_list_truth]
    if len_inferred_values == 1:
        median_list = [median_list]
        yerr_list = [yerr_list]
        label_list_inferred = [label_list_inferred]
        color_list_inferred = [color_list_inferred]
        marker_list_inferred = [marker_list_inferred]

    titles = [
        "$R_{\mathrm{p}}^{\mathrm{xi+}}$",
        "$R_{\mathrm{p}}^{\mathrm{xi-}}$",
        "$R_{\mathrm{p}}^{\mathrm{xe+}}$",
        "$R_{\mathrm{p}}^{\mathrm{xe-}}$",
    ]
    ylabels = [
        "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    ]

    order = [[1, 0], [0, 1], [1, 1], [0, 0]]
    fig = plt.figure(figsize=(12 * 2, 6 * 2))
    for k in range(4):
        i, j = order[k]
        ax = fig.add_subplot(2, 2, j * 2 + i + 1)
        for kk in range(len_true_values):
            ax.plot(
                wavelength,
                truth_list[kk][k],
                color=color_list_truth[kk][k],
                # marker=marker_list_truth[k][j][i],
                # s=150,
                linestyle=ls,
                lw=3,
                # color="0.0",
                # ecolor="0.3",
                # elinewidth=0.8,
                # zorder=3,
                label=label_list_truth[kk][k],
            )
        for kk in range(len_inferred_values):
            if marker_list_inferred[kk][k] != "s":
                ax.errorbar(
                    wavelength,
                    median_list[kk][k],
                    yerr=yerr_list[kk][k],
                    ecolor=color_list_inferred[kk][k],
                    markerfacecolor=color_list_inferred[kk][k],
                    markeredgecolor=color_list_inferred[kk][k],
                    marker=marker_list_inferred[kk][k],
                    markersize=8,
                    linestyle="None",
                    # lw=3,
                    # zorder=5,
                    # alpha=0.7,
                    label=label_list_inferred[kk][k],
                )
            else:
                ax.errorbar(
                    wavelength,
                    median_list[kk][k],
                    yerr=yerr_list[kk][k],
                    ecolor="grey",
                    markerfacecolor=color_list_inferred[kk][k],
                    markeredgecolor="grey",
                    marker=marker_list_inferred[kk][k],
                    markersize=8,
                    linestyle="None",
                    # lw=3,
                    # zorder=5,
                    # alpha=0.7,
                    label=label_list_inferred[kk][k],
                )
        ax.set_ylabel(ylabels[k])
        ax.set_title(titles[k])
        # ax.set_ylim(0.137, 0.163)
        ax.set_ylim(ylim[0], ylim[1])
        # if j == 0:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        if i == 0 and j == 1:
            axLine_i, axLabel_i = ax.get_legend_handles_labels()
            ax.set_xlabel("Wavelength [nm]")
        if i == 1 and j == 1:
            axLine_e, axLabel_e = ax.get_legend_handles_labels()
            ax.set_xlabel("Wavelength [nm]")
    axLine = [axLine_i[0]] + axLine_e
    axLabel = [axLabel_i[0]] + axLabel_e
    # fig.legend(lines[4:], labels[4:], bbox_to_anchor=(1.01, 0.98), loc="upper left")

    # num_items = len(axLine)
    # rows = (num_items - 1) // ncol + 1
    # ordered_indices = [
    #     c + r * ncol
    #     for c in range(ncol)
    #     for r in range(rows)
    #     if c + r * ncol < num_items
    # ]
    # print(axLabel)
    # axLine = [axLine[i] for i in ordered_indices]
    # axLabel = [axLabel[i] for i in ordered_indices]
    # print(axLabel)

    # plt.tick_params(labelsize=16)
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.1)
    fig.legend(
        axLine,
        axLabel,
        bbox_to_anchor=(bbox_to_anchor[0], 0.01),
        loc="upper left",
        ncol=ncol,
    )
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

    rp_over_rs_ingress = 0.150 - 0.005 * np.sin(wavelength * np.pi)
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
    # dc=0 (rp, t1, t2, t3, t4)
    ylabel = [
        "$R_{\mathrm{p}}$ [$R_{\mathrm{s}}$]",
        "$t_{\mathrm{I}}$ [$\mathrm{s}$]",
        "$t_{\mathrm{II}}$ [$\mathrm{s}$]",
        "$t_{\mathrm{III}}$ [$\mathrm{s}$]",
        "$t_{\mathrm{IV}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{i}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{e}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{i}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{e}}$ [$\mathrm{s}$]",
    ]

    posterior_samples = np.load("mcmc_results/model_ecc0_jitter_0/posterior_sample.npz")
    sample_dict, _ = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    median_dict, err_dict = median_yerr(sample_dict)

    median_list = [
        median_dict["rp_over_rs"],
        median_dict["t1"],
        median_dict["t2"],
        median_dict["t3"],
        median_dict["t4"],
    ]
    yerr_list = [
        err_dict["rp_over_rs"],
        err_dict["t1"],
        err_dict["t2"],
        err_dict["t3"],
        err_dict["t4"],
    ]

    truth_list = [[rp_over_rs] * 3, t1, t2, t3, t4]
    label_list_truth = [["True values"] * 3] * 5
    titles = [
        "$e=0$",
        "$e\cos\omega=0.1$, $e\sin\omega=0$",
        "$e\cos\omega=0$, $e\sin\omega=0.1$",
    ]
    color_list_truth = [["black"] * 3] * 5
    marker_list_truth = [["o"] * 3] * 5
    label_list_inferred = [["Inferred values (no noise)"] * 3] * 5
    color_list_inferred = [["black"] * 3] * 5
    marker_list_inferred = [["o"] * 3] * 5
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        filename="sim_rp_contacttimes_using_ecc0_from_dc0_jitter0.png",
        bbox_to_anchor=[0.3, 0.4],
    )

    posterior_samples_5 = np.load(
        "mcmc_results/model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_5, _ = convert_samples_rptn(posterior_samples_5)
    for param in sample_dict_5:
        sample_dict_5[param] = sample_dict_5[param].reshape((num_samples, *(3, 21)))
    median_dict_5, err_dict_5 = median_yerr(sample_dict_5)

    median_list = [
        [
            median_dict["rp_over_rs"],
            median_dict["t1"],
            median_dict["t2"],
            median_dict["t3"],
            median_dict["t4"],
        ],
        [
            median_dict_5["rp_over_rs"],
            median_dict_5["t1"],
            median_dict_5["t2"],
            median_dict_5["t3"],
            median_dict_5["t4"],
        ],
    ]
    yerr_list = [
        [
            err_dict["rp_over_rs"],
            err_dict["t1"],
            err_dict["t2"],
            err_dict["t3"],
            err_dict["t4"],
        ],
        [
            err_dict_5["rp_over_rs"],
            err_dict_5["t1"],
            err_dict_5["t2"],
            err_dict_5["t3"],
            err_dict_5["t4"],
        ],
    ]
    truth_list = [[rp_over_rs] * 3, t1, t2, t3, t4]
    label_list_truth = [["True values"] * 3] * 5
    titles = [
        "$e=0$",
        "$e\cos\omega=0.1$, $e\sin\omega=0$",
        "$e\cos\omega=0$, $e\sin\omega=0.1$",
    ]
    color_list_truth = [["black"] * 3] * 5
    marker_list_truth = [["o"] * 3] * 5

    label_list_inferred = [
        [["Inferred values (no noise)"] * 3] * 5,
        [["Inferred values (with noise)"] * 3] * 5,
    ]
    color_list_inferred = [[["black"] * 3] * 5, [["grey"] * 3] * 5]
    marker_list_inferred = [[["o"] * 3] * 5, [["d"] * 3] * 5]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        len_true_values=1,
        len_inferred_values=2,
        dir_output=dir_output,
        filename="sim_rp_contacttimes_using_ecc0_from_dc0_jitter0_jitter25.png",
        bbox_to_anchor=[0.15, 0.25],
    )

    # -----------------------------------------------------------------------
    # dc=0 (dcX_i, dcY_i. dcX_e, dcY_e)
    ylabel = [
        "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{i,yi}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{e,ye}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    ]

    posterior_samples = np.load("mcmc_results/model_ecc0_jitter_0/posterior_sample.npz")
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    sample_dict_dc_true = convert_samples_ctv(
        sample_dict, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true, err_dict_dc_true = median_yerr(sample_dict_dc_true)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print("ecc", np.array([[0]] * 3))
    print("omega", np.array([[0]] * 3))
    print(
        "cosi", np.mean(orbit_median_dict["cosi"], axis=-1), orbit_median_dict["cosi"]
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.array([[0]] * 3),
        np.array([[0]] * 3),
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    median_list = [
        [
            median_dict_dc_inferred["dc_over_rs_X_ingress"],
            median_dict_dc_inferred["dc_over_rs_Y_ingress"],
            median_dict_dc_inferred["dc_over_rs_X_egress"],
            median_dict_dc_inferred["dc_over_rs_Y_egress"],
        ],
        [
            median_dict_dc_true["dc_over_rs_X_ingress"],
            median_dict_dc_true["dc_over_rs_Y_ingress"],
            median_dict_dc_true["dc_over_rs_X_egress"],
            median_dict_dc_true["dc_over_rs_Y_egress"],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["dc_over_rs_X_ingress"],
            err_dict_dc_inferred["dc_over_rs_Y_ingress"],
            err_dict_dc_inferred["dc_over_rs_X_egress"],
            err_dict_dc_inferred["dc_over_rs_Y_egress"],
        ],
        [
            err_dict_dc_true["dc_over_rs_X_ingress"],
            err_dict_dc_true["dc_over_rs_Y_ingress"],
            err_dict_dc_true["dc_over_rs_X_egress"],
            err_dict_dc_true["dc_over_rs_Y_egress"],
        ],
    ]

    truth_list = [
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
    ]
    label_list_truth = [["True values"] * 3] * 4
    titles = [
        "$e=0$",
        "$e\cos\omega=0.1$, $e\sin\omega=0$",
        "$e\cos\omega=0$, $e\sin\omega=0.1$",
    ]
    color_list_truth = [["black"] * 3] * 4
    marker_list_truth = [["o"] * 3] * 4
    label_list_inferred = [
        [["Inferred values (using fitted orbit)"] * 3] * 4,
        [["Inferred values (using true orbit)"] * 3] * 4,
    ]
    color_list_inferred = [[["black"] * 3] * 4, [["none"] * 3] * 4]
    marker_list_inferred = [[["o"] * 3] * 4, [["s"] * 3] * 4]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        len_true_values=1,
        len_inferred_values=2,
        dir_output=dir_output,
        filename="sim_dc_using_ecc0_from_dc0_jitter0.png",
        bbox_to_anchor=[0.1, 0.25],
    )

    posterior_samples_25 = np.load(
        "mcmc_results/model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples_25)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    median_dict_25, err_dict_25 = median_yerr(sample_dict_25)
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    sample_dict_dc_true_25 = convert_samples_ctv(
        sample_dict_25, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true_25, err_dict_dc_true_25 = median_yerr(sample_dict_dc_true_25)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print("ecc", np.array([[0]] * 3))
    print("omega", np.array([[0]] * 3))
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.array([[0]] * 3),
        np.array([[0]] * 3),
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            median_dict_dc_inferred["dc_over_rs_X_ingress"],
            median_dict_dc_inferred["dc_over_rs_Y_ingress"],
            median_dict_dc_inferred["dc_over_rs_X_egress"],
            median_dict_dc_inferred["dc_over_rs_Y_egress"],
        ],
        [
            median_dict_dc_true["dc_over_rs_X_ingress"],
            median_dict_dc_true["dc_over_rs_Y_ingress"],
            median_dict_dc_true["dc_over_rs_X_egress"],
            median_dict_dc_true["dc_over_rs_Y_egress"],
        ],
        [
            median_dict_dc_inferred_25["dc_over_rs_X_ingress"],
            median_dict_dc_inferred_25["dc_over_rs_Y_ingress"],
            median_dict_dc_inferred_25["dc_over_rs_X_egress"],
            median_dict_dc_inferred_25["dc_over_rs_Y_egress"],
        ],
        [
            median_dict_dc_true_25["dc_over_rs_X_ingress"],
            median_dict_dc_true_25["dc_over_rs_Y_ingress"],
            median_dict_dc_true_25["dc_over_rs_X_egress"],
            median_dict_dc_true_25["dc_over_rs_Y_egress"],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["dc_over_rs_X_ingress"],
            err_dict_dc_inferred["dc_over_rs_Y_ingress"],
            err_dict_dc_inferred["dc_over_rs_X_egress"],
            err_dict_dc_inferred["dc_over_rs_Y_egress"],
        ],
        [
            err_dict_dc_true["dc_over_rs_X_ingress"],
            err_dict_dc_true["dc_over_rs_Y_ingress"],
            err_dict_dc_true["dc_over_rs_X_egress"],
            err_dict_dc_true["dc_over_rs_Y_egress"],
        ],
        [
            err_dict_dc_inferred_25["dc_over_rs_X_ingress"],
            err_dict_dc_inferred_25["dc_over_rs_Y_ingress"],
            err_dict_dc_inferred_25["dc_over_rs_X_egress"],
            err_dict_dc_inferred_25["dc_over_rs_Y_egress"],
        ],
        [
            err_dict_dc_true_25["dc_over_rs_X_ingress"],
            err_dict_dc_true_25["dc_over_rs_Y_ingress"],
            err_dict_dc_true_25["dc_over_rs_X_egress"],
            err_dict_dc_true_25["dc_over_rs_Y_egress"],
        ],
    ]

    truth_list = [
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
        0 * np.ones_like(t1),
    ]
    label_list_truth = [["True values"] * 3] * 4
    titles = [
        "$e=0$",
        "$e\cos\omega=0.1$, $e\sin\omega=0$",
        "$e\cos\omega=0$, $e\sin\omega=0.1$",
    ]
    color_list_truth = [["dodgerblue"] * 3] * 4
    marker_list_truth = [["o"] * 3] * 4
    label_list_inferred = [
        [["Inferred values (using fitted orbit, no noise)"] * 3] * 4,
        [["Inferred values (using true orbit, no noise)"] * 3] * 4,
        [["Inferred values (using fitted orbit, with noise)"] * 3] * 4,
        [["Inferred values (using true orbit, with noise)"] * 3] * 4,
    ]
    color_list_inferred = [
        [["black"] * 3] * 4,
        [["none"] * 3] * 4,
        [["grey"] * 3] * 4,
        [["none"] * 3] * 4,
    ]
    marker_list_inferred = [
        [["o"] * 3] * 4,
        [["s"] * 3] * 4,
        [["d"] * 3] * 4,
        [["s"] * 3] * 4,
    ]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        len_true_values=1,
        len_inferred_values=4,
        dir_output=dir_output,
        filename="sim_dc_using_ecc0_from_dc0_jitter0_jitter25.png",
        bbox_to_anchor=[0.01, 0.01],
    )

    # -----------------------------------------------------------------------
    # dc0 (rpxi+, rpxi-, rpxe+, rpxe-)
    ylabel = [
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{i,yi}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]",
        # "$\Delta c_{\mathrm{e,ye}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    ]

    posterior_samples = np.load("mcmc_results/model_ecc0_jitter_0/posterior_sample.npz")
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print("ecc", np.array([[0]] * 3))
    print("omega", np.array([[0]] * 3))
    print(
        "cosi", np.mean(orbit_median_dict["cosi"], axis=-1), orbit_median_dict["cosi"]
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.array([[0]] * 3),
        np.array([[0]] * 3),
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    posterior_samples_25 = np.load(
        "mcmc_results/model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples_25)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    median_dict_25, err_dict_25 = median_yerr(sample_dict_25)
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print("ecc", np.array([[0]] * 3))
    print("omega", np.array([[0]] * 3))
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.array([[0]] * 3),
        np.array([[0]] * 3),
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            median_dict_dc_inferred["rp_xip"],
            median_dict_dc_inferred["rp_xin"],
            median_dict_dc_inferred["rp_xep"],
            median_dict_dc_inferred["rp_xen"],
        ],
        [
            median_dict_dc_inferred_25["rp_xip"],
            median_dict_dc_inferred_25["rp_xin"],
            median_dict_dc_inferred_25["rp_xep"],
            median_dict_dc_inferred_25["rp_xen"],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["rp_xip"],
            err_dict_dc_inferred["rp_xin"],
            err_dict_dc_inferred["rp_xep"],
            err_dict_dc_inferred["rp_xen"],
        ],
        [
            err_dict_dc_inferred_25["rp_xip"],
            err_dict_dc_inferred_25["rp_xin"],
            err_dict_dc_inferred_25["rp_xep"],
            err_dict_dc_inferred_25["rp_xen"],
        ],
    ]

    truth_list = [
        rp_over_rs * np.ones_like(t1),
        rp_over_rs * np.ones_like(t1),
        rp_over_rs * np.ones_like(t1),
        rp_over_rs * np.ones_like(t1),
    ]
    label_list_truth = [["True values"] * 3] * 4
    titles = [
        "$e=0$",
        "$e\cos\omega=0.1$, $e\sin\omega=0$",
        "$e\cos\omega=0$, $e\sin\omega=0.1$",
    ]
    color_list_truth = [["black"] * 3] * 4
    marker_list_truth = [["o"] * 3] * 4
    label_list_inferred = [
        [["Inferred values (using fitted orbit, no noise)"] * 3] * 4,
        [["Inferred values (using fitted orbit, with noise)"] * 3] * 4,
    ]
    color_list_inferred = [[["black"] * 3] * 4, [["grey"] * 3] * 4]
    marker_list_inferred = [[["o"] * 3] * 4, [["d"] * 3] * 4]

    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        len_true_values=1,
        len_inferred_values=2,
        dir_output=dir_output,
        filename="sim_rp4_using_ecc0_from_dc0_jitter0_jitter25.png",
        bbox_to_anchor=[0.01, 0.01],
    )

    # -----------------------------------------------------------------------
    # dc (rp, ti, te, tau)
    ylabel = [
        "$R_{\mathrm{p}}$ [$R_{\mathrm{s}}$]",
        "$t_{\mathrm{i}}$ [$\mathrm{s}$]",
        "$t_{\mathrm{e}}$ [$\mathrm{s}$]",
        r"$\tau_{\mathrm{i}}$ [$\mathrm{s}$]",
        r"$\tau_{\mathrm{e}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{i}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{e}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{i}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{e}}$ [$\mathrm{s}$]",
    ]

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, _ = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    median_dict, err_dict = median_yerr(sample_dict)

    posterior_samples = np.load(
        "mcmc_results/dc_model_eccfree_jitter_0/posterior_sample.npz"
    )
    sample_dict_f, _ = convert_samples_rptn(posterior_samples, eccfree=True)
    for param in sample_dict_f:
        sample_dict_f[param] = sample_dict_f[param].reshape((num_samples, *(3, 21)))
    median_dict_f, err_dict_f = median_yerr(sample_dict_f)

    median_list = [
        np.array([median_dict["rp_over_rs"][0], median_dict_f["rp_over_rs"][0]]),
        np.array([median_dict["ti"][0], median_dict_f["ti"][0]]),
        np.array([median_dict["te"][0], median_dict_f["te"][0]]),
        np.array([median_dict["taue"][0], median_dict_f["taue"][0]]),
        np.array([median_dict["taue"][0], median_dict_f["taue"][0]]),
    ]
    yerr_list = [
        np.stack(
            [err_dict["rp_over_rs"][:, 0], err_dict_f["rp_over_rs"][:, 0]], axis=1
        ),
        np.stack([err_dict["ti"][:, 0], err_dict_f["ti"][:, 0]], axis=1),
        np.stack([err_dict["te"][:, 0], err_dict_f["te"][:, 0]], axis=1),
        np.stack([err_dict["taui"][:, 0], err_dict_f["taui"][:, 0]], axis=1),
        np.stack([err_dict["taue"][:, 0], err_dict_f["taue"][:, 0]], axis=1),
    ]

    truth_list = [
        [
            [rp_over_rs] * 2,
            [(t1[0] + t2[0]) / 2] * 2,
            [(t3[0] + t4[0]) / 2] * 2,
            [t2[0] - t1[0]] * 2,
            [t4[0] - t3[0]] * 2,
        ],
        [
            [rp_over_rs] * 2,
            [(t1_dc[0] + t2_dc[0]) / 2] * 2,
            [(t3_dc[0] + t4_dc[0]) / 2] * 2,
            [t4_dc[0] - t3_dc[0], t2_dc[0] - t1_dc[0]],
            [t2_dc[0] - t1_dc[0], t4_dc[0] - t3_dc[0]],
        ],
        [
            [rp_over_rs] * 2,
            [(t1_dc[0] + t2_dc[0]) / 2] * 2,
            [(t3_dc[0] + t4_dc[0]) / 2] * 2,
            [t2_dc[0] - t1_dc[0]] * 2,
            [t4_dc[0] - t3_dc[0]] * 2,
        ],
    ]
    label_list_truth = [
        [["Symmetrical"] * 2] * 5,
        [["True values (ingress)"] * 2] * 5,
        [["True values (egress)"] * 2] * 5,
    ]
    titles = [
        "Using circular orbit",
        "Using elliptical orbit",
    ]
    color_list_truth = [
        [["grey"] * 2] * 5,
        [
            ["grey", "grey"],
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
            ["orange", "dodgerblue"],
            ["dodgerblue", "orange"],
        ],
        [
            ["grey", "grey"],
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
        ],
    ]

    marker_list_truth = [[["o"] * 2] * 5] * 3
    label_list_inferred = [["Inferred values (no noise)"] * 2] * 5
    color_list_inferred = [["black"] * 2] * 5
    marker_list_inferred = [["o"] * 2] * 5
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=3,
        len_inferred_values=1,
        filename="sim_rp_titetau_using_ecc0eccfree_from_dc_jitter0.png",
        bbox_to_anchor=[0.01, 0.01],
    )

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, _ = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    median_dict_25, err_dict_25 = median_yerr(sample_dict_25)

    posterior_samples = np.load(
        "mcmc_results/dc_model_eccfree_jitter_000025/posterior_sample.npz"
    )
    sample_dict_f_25, _ = convert_samples_rptn(posterior_samples, eccfree=True)
    for param in sample_dict_f_25:
        sample_dict_f_25[param] = sample_dict_f_25[param].reshape(
            (num_samples, *(3, 21))
        )
    median_dict_f_25, err_dict_f_25 = median_yerr(sample_dict_f_25)

    median_list = [
        [
            np.array([median_dict["rp_over_rs"][0], median_dict_f["rp_over_rs"][0]]),
            np.array([median_dict["ti"][0], median_dict_f["ti"][0]]),
            np.array([median_dict["te"][0], median_dict_f["te"][0]]),
            np.array([median_dict["taui"][0], median_dict_f["taui"][0]]),
            np.array([median_dict["taue"][0], median_dict_f["taue"][0]]),
        ],
        [
            np.array(
                [median_dict_25["rp_over_rs"][0], median_dict_f_25["rp_over_rs"][0]]
            ),
            np.array([median_dict_25["ti"][0], median_dict_f_25["ti"][0]]),
            np.array([median_dict_25["te"][0], median_dict_f_25["te"][0]]),
            np.array([median_dict_25["taui"][0], median_dict_f_25["taui"][0]]),
            np.array([median_dict_25["taue"][0], median_dict_f_25["taue"][0]]),
        ],
    ]
    yerr_list = [
        [
            np.stack(
                [err_dict["rp_over_rs"][:, 0], err_dict_f["rp_over_rs"][:, 0]], axis=1
            ),
            np.stack([err_dict["ti"][:, 0], err_dict_f["ti"][:, 0]], axis=1),
            np.stack([err_dict["te"][:, 0], err_dict_f["te"][:, 0]], axis=1),
            np.stack([err_dict["taui"][:, 0], err_dict_f["taui"][:, 0]], axis=1),
            np.stack([err_dict["taue"][:, 0], err_dict_f["taue"][:, 0]], axis=1),
        ],
        [
            np.stack(
                [err_dict_25["rp_over_rs"][:, 0], err_dict_f_25["rp_over_rs"][:, 0]],
                axis=1,
            ),
            np.stack([err_dict_25["ti"][:, 0], err_dict_f_25["ti"][:, 0]], axis=1),
            np.stack([err_dict_25["te"][:, 0], err_dict_f_25["te"][:, 0]], axis=1),
            np.stack([err_dict_25["taui"][:, 0], err_dict_f_25["taui"][:, 0]], axis=1),
            np.stack([err_dict_25["taue"][:, 0], err_dict_f_25["taue"][:, 0]], axis=1),
        ],
    ]

    label_list_inferred = [
        [["Inferred values (no noise)"] * 2] * 5,
        [["Inferred values (with noise)"] * 2] * 5,
    ]
    color_list_inferred = [[["black"] * 2] * 5, [["grey"] * 2] * 5]
    marker_list_inferred = [[["o"] * 2] * 5, [["d"] * 2] * 5]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=3,
        len_inferred_values=2,
        filename="sim_rp_titetau_using_ecc0eccfree_from_dc_jitter0_jitter25.png",
        bbox_to_anchor=[0.01, 0.01],
    )

    # -----------------------------------------------------------------------
    # dc (rp, ti, te, tau) ecc0 only

    ylabel = [
        "$R_{\mathrm{p}}$ [$R_{\mathrm{s}}$]",
        "$t_{\mathrm{i}}$ [$\mathrm{s}$]",
        "$t_{\mathrm{e}}$ [$\mathrm{s}$]",
        r"$\tau_{\mathrm{i}}, \tau_{\mathrm{e}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{i}}$ [$\mathrm{s}$]",
        # "$t_{\mathrm{e}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{i}}$ [$\mathrm{s}$]",
        # r"$\tau_{\mathrm{e}}$ [$\mathrm{s}$]",
    ]

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, _ = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    median_dict, err_dict = median_yerr(sample_dict)

    truth_list = [
        [
            [rp_over_rs],
            [(t1[0] + t2[0]) / 2],
            [(t3[0] + t4[0]) / 2],
            [t2[0] - t1[0]],
        ],
        [
            [rp_over_rs],
            [(t1_dc[0] + t2_dc[0]) / 2],
            [(t3_dc[0] + t4_dc[0]) / 2],
            [t4_dc[0] - t3_dc[0]],
        ],
        [
            [rp_over_rs],
            [(t1_dc[0] + t2_dc[0]) / 2],
            [(t3_dc[0] + t4_dc[0]) / 2],
            [t2_dc[0] - t1_dc[0]],
        ],
    ]
    label_list_truth = [
        [["Symmetrical"]] * 4,
        [["True values (ingress)"]] * 4,
        [["True values (egress)"]] * 4,
    ]
    titles = [
        "",
    ]
    color_list_truth = [
        [["grey"]] * 4,
        [
            ["grey"],
            ["dodgerblue"],
            ["orange"],
            ["orange"],
        ],
        [
            ["grey"],
            ["dodgerblue"],
            ["orange"],
            ["dodgerblue"],
        ],
    ]
    marker_list_truth = [[["o"]] * 4] * 3

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, _ = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    median_dict_25, err_dict_25 = median_yerr(sample_dict_25)

    median_list = [
        [
            np.array([median_dict["rp_over_rs"][0]]),
            np.array([median_dict["ti"][0]]),
            np.array([median_dict["te"][0]]),
            np.array([median_dict["taui"][0]]),
        ],
        [
            np.array([median_dict_25["rp_over_rs"][0]]),
            np.array([median_dict_25["ti"][0]]),
            np.array([median_dict_25["te"][0]]),
            np.array([median_dict_25["taui"][0]]),
        ],
    ]
    yerr_list = [
        [
            np.array([err_dict["rp_over_rs"][:, 0]]),
            np.array([err_dict["ti"][:, 0]]),
            np.array([err_dict["te"][:, 0]]),
            np.array([err_dict["taui"][:, 0]]),
        ],
        [
            np.array([err_dict_25["rp_over_rs"][:, 0]]),
            np.array([err_dict_25["ti"][:, 0]]),
            np.array([err_dict_25["te"][:, 0]]),
            np.array([err_dict_25["taui"][:, 0]]),
        ],
    ]

    label_list_inferred = [
        [["Inferred values (no noise)"]] * 4,
        [["Inferred values (with noise)"]] * 4,
    ]
    color_list_inferred = [[["black"]] * 4, [["grey"]] * 4]
    marker_list_inferred = [[["o"]] * 4, [["d"]] * 4]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=3,
        len_inferred_values=2,
        filename="sim_rp_titetau_using_ecc0_from_dc_jitter0_jitter25.png",
        bbox_to_anchor=[-0.03, 0.01],
    )

    # -----------------------------------------------------------------------
    # dc (dcxi, dcY, dcxe, dcY)
    ylabel = [
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    ]

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    sample_dict_dc_true = convert_samples_ctv(
        sample_dict, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true, err_dict_dc_true = median_yerr(sample_dict_dc_true)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict["ecosw"] ** 2 + orbit_median_dict["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict["ecosw"], orbit_median_dict["esinw"]),
    )
    print(
        "cosi", np.mean(orbit_median_dict["cosi"], axis=-1), orbit_median_dict["cosi"]
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    posterior_samples = np.load(
        "mcmc_results/dc_model_eccfree_jitter_0/posterior_sample.npz"
    )
    sample_dict_f, orbit_dict_f = convert_samples_rptn(posterior_samples, eccfree=True)
    for param in sample_dict_f:
        sample_dict_f[param] = sample_dict_f[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_f:
        orbit_dict_f[param] = orbit_dict_f[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_f, orbit_err_dict_f = median_yerr(orbit_dict_f)

    sample_dict_dc_true_f = convert_samples_ctv(
        sample_dict_f, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true_f, err_dict_dc_true_f = median_yerr(sample_dict_dc_true_f)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict_f["a_over_rs"], axis=-1),
        orbit_median_dict_f["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_f["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_f["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict_f["ecosw"] ** 2 + orbit_median_dict_f["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_f["ecosw"], axis=-1),
            np.mean(orbit_median_dict_f["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_f["ecosw"], orbit_median_dict_f["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_f["cosi"], axis=-1),
        orbit_median_dict_f["cosi"],
    )
    print("t0", np.mean(orbit_median_dict_f["t0"], axis=-1), orbit_median_dict_f["t0"])
    sample_dict_dc_inferred_f = convert_samples_ctv(
        sample_dict_f,
        period,
        np.mean(orbit_median_dict_f["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_f["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_f["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_f["ecosw"], axis=-1),
            np.mean(orbit_median_dict_f["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_f["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_f["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_f, err_dict_dc_inferred_f = median_yerr(
        sample_dict_dc_inferred_f
    )

    median_list = [
        [
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xe_egress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_egress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
        [
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_true_f["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_true_f["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xe_egress"][0],
                    median_dict_dc_true_f["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_egress"][0],
                    median_dict_dc_true_f["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
    ]
    yerr_list = [
        [
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
        [
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
    ]

    truth_list = [
        [
            [dc_over_rs_xi_ingress[0]] * 2,
            [dc_over_rs_Y_egress, dc_over_rs_Y_ingress],
            [dc_over_rs_xe_egress[0]] * 2,
            [dc_over_rs_Y_ingress, dc_over_rs_Y_egress],
        ],
        [
            [dc_over_rs_xi_ingress[0]] * 2,
            [dc_over_rs_Y_ingress] * 2,
            [dc_over_rs_xe_egress[0]] * 2,
            [dc_over_rs_Y_egress] * 2,
        ],
    ]
    label_list_truth = [
        [["True values (ingress)"] * 2] * 4,
        [["True values (egress)"] * 2] * 4,
    ]
    titles = [
        "Using circular orbit",
        "Using elliptical orbit",
    ]
    color_list_truth = [
        [
            ["dodgerblue", "dodgerblue"],
            ["orange", "dodgerblue"],
            ["orange", "orange"],
            ["dodgerblue", "orange"],
        ],
        [
            ["dodgerblue", "dodgerblue"],
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
            ["orange", "orange"],
        ],
    ]
    marker_list_truth = [[["o"] * 2] * 4, [["o"] * 2] * 4]

    label_list_inferred = [
        [["Inferred values (using fitted orbit, no noise)"] * 2] * 4,
        [["Inferred values (using true orbit, no noise)"] * 2] * 4,
    ]
    color_list_inferred = [[["black"] * 2] * 4, [["none"] * 2] * 4]
    marker_list_inferred = [[["o"] * 2] * 4, [["s"] * 2] * 4]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=2,
        len_inferred_values=2,
        filename="sim_dc_using_ecc0eccfree_from_dc_jitter0.png",
        bbox_to_anchor=[0.01, 0.01],
        ncol=2,
    )

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    sample_dict_dc_true_25 = convert_samples_ctv(
        sample_dict_25, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true_25, err_dict_dc_true_25 = median_yerr(sample_dict_dc_true_25)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_25["ecosw"] ** 2 + orbit_median_dict_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_25["ecosw"], orbit_median_dict_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    posterior_samples = np.load(
        "mcmc_results/dc_model_eccfree_jitter_000025/posterior_sample.npz"
    )
    sample_dict_f_25, orbit_dict_f_25 = convert_samples_rptn(
        posterior_samples, eccfree=True
    )
    for param in sample_dict_f_25:
        sample_dict_f_25[param] = sample_dict_f_25[param].reshape(
            (num_samples, *(3, 21))
        )
    for param in orbit_dict_f_25:
        orbit_dict_f_25[param] = orbit_dict_f_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_f_25, orbit_err_dict_f_25 = median_yerr(orbit_dict_f_25)

    sample_dict_dc_true_f_25 = convert_samples_ctv(
        sample_dict_f_25, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true_f_25, err_dict_dc_true_f_25 = median_yerr(
        sample_dict_dc_true_f_25
    )
    print(
        "a_over_rs",
        np.mean(orbit_median_dict_f_25["a_over_rs"], axis=-1),
        orbit_median_dict_f_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_f_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_f_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_f_25["ecosw"] ** 2 + orbit_median_dict_f_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_f_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_f_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_f_25["ecosw"], orbit_median_dict_f_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_f_25["cosi"], axis=-1),
        orbit_median_dict_f_25["cosi"],
    )
    print(
        "t0",
        np.mean(orbit_median_dict_f_25["t0"], axis=-1),
        orbit_median_dict_f_25["t0"],
    )
    sample_dict_dc_inferred_f_25 = convert_samples_ctv(
        sample_dict_f_25,
        period,
        np.mean(orbit_median_dict_f_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_f_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_f_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_f_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_f_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_f_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_f_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_f_25, err_dict_dc_inferred_f_25 = median_yerr(
        sample_dict_dc_inferred_f_25
    )

    median_list = [
        [
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xe_egress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_egress"][0],
                    median_dict_dc_inferred_f["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
        [
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_true_f["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_true_f["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xe_egress"][0],
                    median_dict_dc_true_f["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_egress"][0],
                    median_dict_dc_true_f["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
        [
            np.array(
                [
                    median_dict_dc_inferred_25["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_inferred_f_25["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred_25["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_inferred_f_25["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred_25["dc_over_rs_xe_egress"][0],
                    median_dict_dc_inferred_f_25["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred_25["dc_over_rs_Y_egress"][0],
                    median_dict_dc_inferred_f_25["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
        # [
        #     np.array(
        #         [
        #             median_dict_dc_true_25["dc_over_rs_xi_ingress"][0],
        #             median_dict_dc_true_f_25["dc_over_rs_xi_ingress"][0],
        #         ]
        #     ),
        #     np.array(
        #         [
        #             median_dict_dc_true_25["dc_over_rs_yi_ingress"][0],
        #             median_dict_dc_true_f_25["dc_over_rs_yi_ingress"][0],
        #         ]
        #     ),
        #     np.array(
        #         [
        #             median_dict_dc_true_25["dc_over_rs_xe_egress"][0],
        #             median_dict_dc_true_f_25["dc_over_rs_xe_egress"][0],
        #         ]
        #     ),
        #     np.array(
        #         [
        #             median_dict_dc_true_25["dc_over_rs_ye_egress"][0],
        #             median_dict_dc_true_f_25["dc_over_rs_ye_egress"][0],
        #         ]
        #     ),
        # ],
    ]
    yerr_list = [
        [
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_inferred_f["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
        [
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_true_f["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
        [
            np.stack(
                [
                    err_dict_dc_inferred_25["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_inferred_f_25["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred_25["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_inferred_f_25["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred_25["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_inferred_f_25["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred_25["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_inferred_f_25["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
        # [
        #     np.stack(
        #         [
        #             err_dict_dc_true_25["dc_over_rs_xi_ingress"][:, 0],
        #             err_dict_dc_true_f_25["dc_over_rs_xi_ingress"][:, 0],
        #         ],
        #         axis=1,
        #     ),
        #     np.stack(
        #         [
        #             err_dict_dc_true_25["dc_over_rs_yi_ingress"][:, 0],
        #             err_dict_dc_true_f_25["dc_over_rs_yi_ingress"][:, 0],
        #         ],
        #         axis=1,
        #     ),
        #     np.stack(
        #         [
        #             err_dict_dc_true_25["dc_over_rs_xe_egress"][:, 0],
        #             err_dict_dc_true_f_25["dc_over_rs_xe_egress"][:, 0],
        #         ],
        #         axis=1,
        #     ),
        #     np.stack(
        #         [
        #             err_dict_dc_true_25["dc_over_rs_ye_egress"][:, 0],
        #             err_dict_dc_true_f_25["dc_over_rs_ye_egress"][:, 0],
        #         ],
        #         axis=1,
        #     ),
        # ],
    ]

    label_list_inferred = [
        [["Inferred values (using fitted orbit, no noise)"] * 2] * 4,
        [["Inferred values (using true orbit, no noise)"] * 2] * 4,
        [["Inferred values (using fitted orbit, with noise)"] * 2] * 4,
        # [["Inferred values (using true orbit, with noise)"] * 2] * 4,
    ]
    color_list_inferred = [
        [["black"] * 2] * 4,
        [["none"] * 2] * 4,
        [["grey"] * 2] * 4,
        # [["none"] * 2] * 4,
    ]
    marker_list_inferred = [
        [["o"] * 2] * 4,
        [["s"] * 2] * 4,
        [["d"] * 2] * 4,
        # [["s"] * 2] * 4,
    ]
    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=2,
        len_inferred_values=3,
        filename="sim_dc_using_ecc0eccfree_from_dc_jitter0_jitter25.png",
        bbox_to_anchor=[0.01, 0.01],
        ncol=2,
    )

    # -----------------------------------------------------------------------
    # dc (dcxi, dcY, dcxe, dcY)
    ylabel = [
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        # "$\Delta c_{\mathrm{X}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        "$\Delta c_{\mathrm{i,xi}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(ingress)}}$",
        "$\Delta c_{\mathrm{e,xe}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        "$\Delta c_{\mathrm{Y}}$ [$R_{\mathrm{s}}$]\n$^{\mathrm{(egress)}}$",
        # "$R_{\mathrm{p}}^{\mathrm{xi+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xi-}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe+}}$ [$R_{\mathrm{s}}$]",
        # "$R_{\mathrm{p}}^{\mathrm{xe-}}$ [$R_{\mathrm{s}}$]",
    ]

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    sample_dict_dc_true = convert_samples_ctv(
        sample_dict, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true, err_dict_dc_true = median_yerr(sample_dict_dc_true)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict["ecosw"] ** 2 + orbit_median_dict["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict["ecosw"], orbit_median_dict["esinw"]),
    )
    print(
        "cosi", np.mean(orbit_median_dict["cosi"], axis=-1), orbit_median_dict["cosi"]
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    sample_dict_dc_true_25 = convert_samples_ctv(
        sample_dict_25, period, a_over_rs, ecc, omega, cosi, t0
    )
    median_dict_dc_true_25, err_dict_dc_true_25 = median_yerr(sample_dict_dc_true_25)
    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_25["ecosw"] ** 2 + orbit_median_dict_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_25["ecosw"], orbit_median_dict_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_inferred_25["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_inferred_25["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_xe_egress"][0],
                    median_dict_dc_inferred_25["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_inferred["dc_over_rs_Y_egress"][0],
                    median_dict_dc_inferred_25["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
        [
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xi_ingress"][0],
                    median_dict_dc_true_25["dc_over_rs_xi_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_ingress"][0],
                    median_dict_dc_true_25["dc_over_rs_Y_ingress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_xe_egress"][0],
                    median_dict_dc_true_25["dc_over_rs_xe_egress"][0],
                ]
            ),
            np.array(
                [
                    median_dict_dc_true["dc_over_rs_Y_egress"][0],
                    median_dict_dc_true_25["dc_over_rs_Y_egress"][0],
                ]
            ),
        ],
    ]
    yerr_list = [
        [
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_inferred_25["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_inferred_25["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_inferred_25["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_inferred["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_inferred_25["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
        [
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xi_ingress"][:, 0],
                    err_dict_dc_true_25["dc_over_rs_xi_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_ingress"][:, 0],
                    err_dict_dc_true_25["dc_over_rs_Y_ingress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_xe_egress"][:, 0],
                    err_dict_dc_true_25["dc_over_rs_xe_egress"][:, 0],
                ],
                axis=1,
            ),
            np.stack(
                [
                    err_dict_dc_true["dc_over_rs_Y_egress"][:, 0],
                    err_dict_dc_true_25["dc_over_rs_Y_egress"][:, 0],
                ],
                axis=1,
            ),
        ],
    ]

    label_list_inferred = [
        [
            [
                "Inferred values (using fitted orbit, no noise)",
                "Inferred values (using fitted orbit, with noise)",
            ]
        ]
        * 4,
        [
            [
                "Inferred values (using true orbit, no noise)",
                "Inferred values (using true orbit, with noise)",
            ]
        ]
        * 4,
    ]
    color_list_inferred = [
        [["black", "grey"]] * 4,
        [["none", "none"]] * 4,
    ]
    marker_list_inferred = [
        [["o", "d"]] * 4,
        [["s", "s"]] * 4,
    ]

    truth_list = [
        [
            [dc_over_rs_xi_ingress[0]] * 2,
            [dc_over_rs_Y_egress] * 2,
            [dc_over_rs_xe_egress[0]] * 2,
            [dc_over_rs_Y_ingress] * 2,
        ],
        [
            [dc_over_rs_xi_ingress[0]] * 2,
            [dc_over_rs_Y_ingress] * 2,
            [dc_over_rs_xe_egress[0]] * 2,
            [dc_over_rs_Y_egress] * 2,
        ],
    ]
    label_list_truth = [
        [["True values (ingress)"] * 2] * 4,
        [["True values (egress)"] * 2] * 4,
    ]
    titles = [
        "No noise",
        "With noise",
    ]
    color_list_truth = [
        [
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
            ["orange", "orange"],
            ["dodgerblue", "dodgerblue"],
        ],
        [
            ["dodgerblue", "dodgerblue"],
            ["dodgerblue", "dodgerblue"],
            ["orange", "orange"],
            ["orange", "orange"],
        ],
    ]
    marker_list_truth = [[["o"] * 2] * 4, [["o"] * 2] * 4]

    plot_2dlayout(
        titles,
        ylabel,
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=2,
        len_inferred_values=2,
        filename="sim_dc_using_ecc0_from_dc_jitter0_jitter25.png",
        bbox_to_anchor=[0.01, 0.01],
        ncol=2,
    )

    # -----------------------------------------------------------------------
    # dc (rpxi+, rpxi-, rpxe+, rpxe-)
    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict["ecosw"] ** 2 + orbit_median_dict["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict["ecosw"], orbit_median_dict["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict["cosi"], axis=-1),
        orbit_median_dict["cosi"],
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )

    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)
    posterior_samples = np.load(
        "mcmc_results/dc_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_25["ecosw"] ** 2 + orbit_median_dict_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_25["ecosw"], orbit_median_dict_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            median_dict_dc_inferred["rp_xip"][0],
            median_dict_dc_inferred["rp_xin"][0],
            median_dict_dc_inferred["rp_xep"][0],
            median_dict_dc_inferred["rp_xen"][0],
        ],
        [
            median_dict_dc_inferred_25["rp_xip"][0],
            median_dict_dc_inferred_25["rp_xin"][0],
            median_dict_dc_inferred_25["rp_xep"][0],
            median_dict_dc_inferred_25["rp_xen"][0],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["rp_xip"][:, 0],
            err_dict_dc_inferred["rp_xin"][:, 0],
            err_dict_dc_inferred["rp_xep"][:, 0],
            err_dict_dc_inferred["rp_xen"][:, 0],
        ],
        [
            err_dict_dc_inferred_25["rp_xip"][:, 0],
            err_dict_dc_inferred_25["rp_xin"][:, 0],
            err_dict_dc_inferred_25["rp_xep"][:, 0],
            err_dict_dc_inferred_25["rp_xen"][:, 0],
        ],
    ]

    truth_list = [
        rp_xip[0],
        rp_xin[0],
        rp_xep[0],
        rp_xen[0],
    ]
    label_list_truth = ["True values"] * 4
    color_list_truth = ["dodgerblue", "dodgerblue", "orange", "orange"]
    marker_list_truth = ["o"] * 4

    label_list_inferred = [
        ["Inferred values (using fitted orbit, no noise)"] * 4,
        ["Inferred values (using fitted orbit, with noise)"] * 4,
    ]
    color_list_inferred = [["black"] * 4, ["grey"] * 4]
    marker_list_inferred = [["o"] * 4, ["d"] * 4]
    plot_rp4(
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=1,
        len_inferred_values=2,
        filename="sim_rp4_using_ecc0_from_dc_jitter0_jitter25.png",
        bbox_to_anchor=[0.15],
        ncol=2,
        ls="-",
        ylim=[0.1375, 0.1625],
    )

    sample_dict_dc_inferred_25_e = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None]
        + 0.05,
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25_e, err_dict_dc_inferred_25_e = median_yerr(
        sample_dict_dc_inferred_25_e
    )

    median_list = [
        [
            median_dict_dc_inferred_25["rp_xip"][0],
            median_dict_dc_inferred_25["rp_xin"][0],
            median_dict_dc_inferred_25["rp_xep"][0],
            median_dict_dc_inferred_25["rp_xen"][0],
        ],
        [
            median_dict_dc_inferred_25_e["rp_xip"][0],
            median_dict_dc_inferred_25_e["rp_xin"][0],
            median_dict_dc_inferred_25_e["rp_xep"][0],
            median_dict_dc_inferred_25_e["rp_xen"][0],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred_25["rp_xip"][:, 0],
            err_dict_dc_inferred_25["rp_xin"][:, 0],
            err_dict_dc_inferred_25["rp_xep"][:, 0],
            err_dict_dc_inferred_25["rp_xen"][:, 0],
        ],
        [
            err_dict_dc_inferred_25_e["rp_xip"][:, 0],
            err_dict_dc_inferred_25_e["rp_xin"][:, 0],
            err_dict_dc_inferred_25_e["rp_xep"][:, 0],
            err_dict_dc_inferred_25_e["rp_xen"][:, 0],
        ],
    ]

    label_list_inferred = [
        ["Inferred values (using fitted orbit, with noise)"] * 4,
        ["Inferred values (using fitted orbit (e + 0.05), with noise)"] * 4,
    ]
    color_list_inferred = [["black"] * 4, ["grey"] * 4]
    marker_list_inferred = [["o"] * 4, ["s"] * 4]
    plot_rp4(
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=1,
        len_inferred_values=2,
        filename="sim_dc_rp4_jitter25_shift.png",
        bbox_to_anchor=[0.05],
        ncol=2,
    )

    # -----------------------------------------------------------------------
    # catwoman (rpxi+, rpxi-, rpxe+, rpxe-)
    posterior_samples = np.load(
        "mcmc_results/catwoman_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict["ecosw"] ** 2 + orbit_median_dict["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict["ecosw"], orbit_median_dict["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict["cosi"], axis=-1),
        orbit_median_dict["cosi"],
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    posterior_samples = np.load(
        "mcmc_results/catwoman_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_25["ecosw"] ** 2 + orbit_median_dict_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_25["ecosw"], orbit_median_dict_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            median_dict_dc_inferred["rp_xip"][0],
            median_dict_dc_inferred["rp_xin"][0],
            median_dict_dc_inferred["rp_xep"][0],
            median_dict_dc_inferred["rp_xen"][0],
        ],
        [
            median_dict_dc_inferred_25["rp_xip"][0],
            median_dict_dc_inferred_25["rp_xin"][0],
            median_dict_dc_inferred_25["rp_xep"][0],
            median_dict_dc_inferred_25["rp_xen"][0],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["rp_xip"][:, 0],
            err_dict_dc_inferred["rp_xin"][:, 0],
            err_dict_dc_inferred["rp_xep"][:, 0],
            err_dict_dc_inferred["rp_xen"][:, 0],
        ],
        [
            err_dict_dc_inferred_25["rp_xip"][:, 0],
            err_dict_dc_inferred_25["rp_xin"][:, 0],
            err_dict_dc_inferred_25["rp_xep"][:, 0],
            err_dict_dc_inferred_25["rp_xen"][:, 0],
        ],
    ]

    truth_list = [
        rp_over_rs_morning,
        rp_over_rs_evening,
        rp_over_rs_morning,
        rp_over_rs_evening,
    ]
    label_list_truth = [
        "$R_{\mathrm{p}}^{\mathrm{morning}}$",
        "$R_{\mathrm{p}}^{\mathrm{evening}}$",
        "$R_{\mathrm{p}}^{\mathrm{morning}}$",
        "$R_{\mathrm{p}}^{\mathrm{evening}}$",
    ]
    color_list_truth = ["blue", "red", "blue", "red"]
    marker_list_truth = ["o"] * 4

    label_list_inferred = [
        ["Inferred values (using fitted orbit, no noise)"] * 4,
        ["Inferred values (using fitted orbit, with noise)"] * 4,
    ]
    color_list_inferred = [["black"] * 4, ["grey"] * 4]
    marker_list_inferred = [["o"] * 4, ["d"] * 4]
    plot_rp4(
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=1,
        len_inferred_values=2,
        filename="sim_rp4_using_ecc0_from_catwoman_jitter0_jitter25.png",
        bbox_to_anchor=[0.15],
        ncol=2,
        ls="-",
        ylim=[0.1375, 0.1625],
    )

    # -----------------------------------------------------------------------
    # rp_change (rpxi+, rpxi-, rpxe+, rpxe-)
    posterior_samples = np.load(
        "mcmc_results/rpirpe_model_ecc0_jitter_0/posterior_sample.npz"
    )
    sample_dict, orbit_dict = convert_samples_rptn(posterior_samples)
    for param in sample_dict:
        sample_dict[param] = sample_dict[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict:
        orbit_dict[param] = orbit_dict[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict, orbit_err_dict = median_yerr(orbit_dict)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict["a_over_rs"], axis=-1),
        orbit_median_dict["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        ),
        np.sqrt(orbit_median_dict["ecosw"] ** 2 + orbit_median_dict["esinw"] ** 2),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict["ecosw"], orbit_median_dict["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict["cosi"], axis=-1),
        orbit_median_dict["cosi"],
    )
    print("t0", np.mean(orbit_median_dict["t0"], axis=-1), orbit_median_dict["t0"])
    sample_dict_dc_inferred = convert_samples_ctv(
        sample_dict,
        period,
        np.mean(orbit_median_dict["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict["ecosw"], axis=-1),
            np.mean(orbit_median_dict["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred, err_dict_dc_inferred = median_yerr(sample_dict_dc_inferred)

    posterior_samples = np.load(
        "mcmc_results/rpirpe_model_ecc0_jitter_000025/posterior_sample.npz"
    )
    sample_dict_25, orbit_dict_25 = convert_samples_rptn(posterior_samples)
    for param in sample_dict_25:
        sample_dict_25[param] = sample_dict_25[param].reshape((num_samples, *(3, 21)))
    for param in orbit_dict_25:
        orbit_dict_25[param] = orbit_dict_25[param].reshape((num_samples, *(3, 21)))
    orbit_median_dict_25, orbit_err_dict_25 = median_yerr(orbit_dict_25)

    print(
        "a_over_rs",
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1),
        orbit_median_dict_25["a_over_rs"],
    )
    print(
        "ecc",
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        ),
        np.sqrt(
            orbit_median_dict_25["ecosw"] ** 2 + orbit_median_dict_25["esinw"] ** 2
        ),
    )
    print(
        "omega",
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        ),
        np.arctan2(orbit_median_dict_25["ecosw"], orbit_median_dict_25["esinw"]),
    )
    print(
        "cosi",
        np.mean(orbit_median_dict_25["cosi"], axis=-1),
        orbit_median_dict_25["cosi"],
    )
    print(
        "t0", np.mean(orbit_median_dict_25["t0"], axis=-1), orbit_median_dict_25["t0"]
    )
    sample_dict_dc_inferred_25 = convert_samples_ctv(
        sample_dict_25,
        period,
        np.mean(orbit_median_dict_25["a_over_rs"], axis=-1)[:, None],
        np.sqrt(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1) ** 2
            + np.mean(orbit_median_dict_25["esinw"], axis=-1) ** 2
        )[:, None],
        np.arctan2(
            np.mean(orbit_median_dict_25["ecosw"], axis=-1),
            np.mean(orbit_median_dict_25["esinw"], axis=-1),
        )[:, None],
        np.mean(orbit_median_dict_25["cosi"], axis=-1)[:, None],
        np.mean(orbit_median_dict_25["t0"], axis=-1)[:, None],
    )
    median_dict_dc_inferred_25, err_dict_dc_inferred_25 = median_yerr(
        sample_dict_dc_inferred_25
    )

    median_list = [
        [
            median_dict_dc_inferred["rp_xip"][0],
            median_dict_dc_inferred["rp_xin"][0],
            median_dict_dc_inferred["rp_xep"][0],
            median_dict_dc_inferred["rp_xen"][0],
        ],
        [
            median_dict_dc_inferred_25["rp_xip"][0],
            median_dict_dc_inferred_25["rp_xin"][0],
            median_dict_dc_inferred_25["rp_xep"][0],
            median_dict_dc_inferred_25["rp_xen"][0],
        ],
    ]
    yerr_list = [
        [
            err_dict_dc_inferred["rp_xip"][:, 0],
            err_dict_dc_inferred["rp_xin"][:, 0],
            err_dict_dc_inferred["rp_xep"][:, 0],
            err_dict_dc_inferred["rp_xen"][:, 0],
        ],
        [
            err_dict_dc_inferred_25["rp_xip"][:, 0],
            err_dict_dc_inferred_25["rp_xin"][:, 0],
            err_dict_dc_inferred_25["rp_xep"][:, 0],
            err_dict_dc_inferred_25["rp_xen"][:, 0],
        ],
    ]

    truth_list = [
        rp_over_rs_ingress,
        rp_over_rs_ingress,
        rp_over_rs_egress,
        rp_over_rs_egress,
    ]
    label_list_truth = [
        "$R_{\mathrm{p}}^{\mathrm{ingress}}$",
        "$R_{\mathrm{p}}^{\mathrm{ingress}}$",
        "$R_{\mathrm{p}}^{\mathrm{egress}}$",
        "$R_{\mathrm{p}}^{\mathrm{egress}}$",
    ]
    color_list_truth = ["darkblue", "darkblue", "darkorange", "darkorange"]
    marker_list_truth = ["o"] * 4

    label_list_inferred = [
        ["Inferred values (using fitted orbit, no noise)"] * 4,
        ["Inferred values (using fitted orbit, with noise)"] * 4,
    ]
    color_list_inferred = [["black"] * 4, ["grey"] * 4]
    marker_list_inferred = [["o"] * 4, ["d"] * 4]
    plot_rp4(
        wavelength,
        truth_list,
        median_list,
        yerr_list,
        label_list_truth,
        color_list_truth,
        marker_list_truth,
        label_list_inferred,
        color_list_inferred,
        marker_list_inferred,
        dir_output=dir_output,
        len_true_values=1,
        len_inferred_values=2,
        filename="sim_rp4_using_ecc0_from_rpchange_jitter0_jitter25.png",
        bbox_to_anchor=[0.15],
        ncol=2,
        ls="--",
    )
