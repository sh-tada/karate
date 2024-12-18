import matplotlib.pyplot as plt


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
        prediction_median - input,
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
