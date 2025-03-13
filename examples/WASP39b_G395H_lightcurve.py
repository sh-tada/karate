import numpy as np
import xarray as xr


def load_binning(resolution=None):
    stellar_specs = [
        "wasp39_spectra/WASP39-G395H-NRS1-ReductionA.nc",  # NRS1 stellar spec.
        "wasp39_spectra/WASP39-G395H-NRS2-ReductionA.nc",  # NRS2 stellar spec.
    ]

    # Config.
    start_cut = 15  # Starting integration for lcs.

    # Bad columns mask.
    bad_cols = [
        np.array(
            [
                789,
                804,
                805,
                976,
                1063,
                1064,
                1074,
                1214,
                1215,
                1250,
                1308,
                1395,
                1416,
                1442,
                1446,
                1447,
            ],
            dtype=int,
        ),
        np.array(
            [62, 64, 164, 618, 772, 1092, 1096, 1517, 1918, 1969, 1970, 1971, 1088],
            dtype=int,
        ),
    ]

    # Load data
    spec_nrs1 = xr.open_dataset(stellar_specs[0])
    spec_nrs2 = xr.open_dataset(stellar_specs[1])

    time_nrs1 = np.array(spec_nrs1["time_flux"].values)
    time_nrs2 = np.array(spec_nrs2["time_flux"].values)
    time_nrs1 = (time_nrs1 - 59791.0) * 24 * 60 * 60
    time_nrs2 = (time_nrs2 - 59791.0) * 24 * 60 * 60

    wavelength_raw_nrs1 = np.delete(spec_nrs1["wavelength"].values * 1000, bad_cols[0])
    wavelength_raw_nrs2 = np.delete(spec_nrs2["wavelength"].values * 1000, bad_cols[1])
    flux_raw_nrs1 = np.delete(spec_nrs1["flux"].values, bad_cols[0], axis=1).T
    flux_raw_nrs2 = np.delete(spec_nrs2["flux"].values, bad_cols[1], axis=1).T
    # flux = np.concatenate((flux_nrs1, flux_nrs2), axis=0)
    flux_err_raw_nrs1 = np.delete(spec_nrs1["flux_error"].values, bad_cols[0], axis=1).T
    flux_err_raw_nrs2 = np.delete(spec_nrs2["flux_error"].values, bad_cols[1], axis=1).T
    # flux_err = np.concatenate((flux_err_nrs1, flux_err_nrs2), axis=0)

    if resolution is None or resolution == "raw":
        nrs1_mask = wavelength_raw_nrs1 >= 2750
        nrs2_mask = wavelength_raw_nrs2 <= 5100
        # nrs1_mask = wavelength_raw_nrs1 >= 2800
        # nrs2_mask = wavelength_raw_nrs2 <= 5000
        wavelength_nrs1 = wavelength_raw_nrs1[nrs1_mask]
        wavelength_nrs2 = wavelength_raw_nrs2[nrs2_mask]
        flux_nrs1 = flux_raw_nrs1[nrs1_mask]
        flux_nrs2 = flux_raw_nrs2[nrs2_mask]
        flux_err_nrs1 = flux_err_raw_nrs1[nrs1_mask]
        flux_err_nrs2 = flux_err_raw_nrs2[nrs2_mask]
        return (
            time_nrs1,
            time_nrs2,
            wavelength_nrs1,
            wavelength_nrs2,
            flux_nrs1,
            flux_nrs2,
            flux_err_nrs1,
            flux_err_nrs2,
        )

    # Binning from resolution
    elif resolution is not None:
        wavelength_boundary_nrs1, wavelength_boundary_nrs2 = wavelength_boundary(
            resolution, wavelength_raw_nrs1, wavelength_raw_nrs2
        )

        wavelength_nrs1, flux_nrs1, flux_err_nrs1 = binning_weighted(
            start_cut,
            wavelength_raw_nrs1,
            flux_raw_nrs1,
            flux_err_raw_nrs1,
            wavelength_boundary_nrs1,
        )
        wavelength_nrs2, flux_nrs2, flux_err_nrs2 = binning_weighted(
            start_cut,
            wavelength_raw_nrs2,
            flux_raw_nrs2,
            flux_err_raw_nrs2,
            wavelength_boundary_nrs2,
        )

    return (
        time_nrs1,
        time_nrs2,
        wavelength_nrs1,
        wavelength_nrs2,
        flux_nrs1,
        flux_nrs2,
        flux_err_nrs1,
        flux_err_nrs2,
    )


def binning_weighted(
    start_cut, wavelength_raw, flux_raw, flux_err_raw, wavelength_boundary
):
    wavelength = []
    flux = []
    flux_err = []
    for wv_boundary_i in range(len(wavelength_boundary) - 1):
        wv_min = wavelength_raw >= wavelength_boundary[wv_boundary_i]
        wv_max = wavelength_raw < wavelength_boundary[wv_boundary_i + 1]
        wv_bin = wv_min * wv_max
        if np.any(wv_bin):
            flux.append(np.sum(flux_raw[wv_bin], axis=0))
            flux_err.append(np.sqrt(np.sum(flux_err_raw[wv_bin] ** 2, axis=0)))
            # weighted average of wavelength
            wavelength.append(
                np.mean(
                    wavelength_raw[wv_bin]
                    * np.mean(flux_raw[wv_bin, : start_cut + 135], axis=1)
                )
                / np.mean(flux_raw[wv_bin, : start_cut + 135])
            )
        else:
            pass
    return np.array(wavelength), np.array(flux), np.array(flux_err)


def wavelength_boundary(resolution, wavelength_raw_nrs1, wavelength_raw_nrs2):
    wavelength_boundary_nrs1 = []
    wavelength_boundary_nrs1.append(wavelength_raw_nrs1[-1])
    while True:
        new_wv = (
            wavelength_boundary_nrs1[-1] * (2 * resolution - 1) / (2 * resolution + 1)
        )
        if new_wv < 2775:
            # print(wavelength_boundary_nrs1)
            break
        else:
            wavelength_boundary_nrs1.append(new_wv)
    wavelength_boundary_nrs1 = np.array(wavelength_boundary_nrs1)[::-1]
    wavelength_boundary_nrs2 = []
    wavelength_boundary_nrs2.append(wavelength_raw_nrs2[0])
    while True:
        new_wv = (
            wavelength_boundary_nrs2[-1] * (2 * resolution + 1) / (2 * resolution - 1)
        )
        if new_wv > 5050:
            # print(wavelength_boundary_nrs2)
            break
        else:
            wavelength_boundary_nrs2.append(new_wv)
    wavelength_boundary_nrs2 = np.array(wavelength_boundary_nrs2)
    return wavelength_boundary_nrs1, wavelength_boundary_nrs2


def remove_tilt_event(
    time_nrs1, time_nrs2, flux_nrs1, flux_nrs2, flux_err_nrs1, flux_err_nrs2
):

    start_cut = 15  # Starting integration for lcs.
    tilt_start = 269  # Start of tilt integration.
    tilt_end = 272  # End of tilt integration.

    # Divide before and after the tilt event
    t_pre_nrs1 = time_nrs1[start_cut:tilt_start]
    t_pre_nrs2 = time_nrs2[start_cut:tilt_start]
    t_pst_nrs1 = time_nrs1[tilt_end:]
    t_pst_nrs2 = time_nrs2[tilt_end:]
    t_nrs1 = np.concatenate((t_pre_nrs1, t_pst_nrs1))
    t_nrs2 = np.concatenate((t_pre_nrs2, t_pst_nrs2))

    pre_tilt_f_nrs1 = flux_nrs1[:, start_cut:tilt_start]
    pre_tilt_f_nrs2 = flux_nrs2[:, start_cut:tilt_start]
    pst_tilt_f_nrs1 = flux_nrs1[:, tilt_end:]
    pst_tilt_f_nrs2 = flux_nrs2[:, tilt_end:]
    pre_tilt_f_err_nrs1 = flux_err_nrs1[:, start_cut:tilt_start]
    pre_tilt_f_err_nrs2 = flux_err_nrs2[:, start_cut:tilt_start]
    pst_tilt_f_err_nrs1 = flux_err_nrs1[:, tilt_end:]
    pst_tilt_f_err_nrs2 = flux_err_nrs2[:, tilt_end:]
    return (
        [t_nrs1, t_nrs2],
        [pre_tilt_f_nrs1, pre_tilt_f_nrs2],
        [pst_tilt_f_nrs1, pst_tilt_f_nrs2],
        [pre_tilt_f_err_nrs1, pre_tilt_f_err_nrs2],
        [pst_tilt_f_err_nrs1, pst_tilt_f_err_nrs2],
    )


def normalize_by_pretilt(pre_tilt_f, pst_tilt_f, pre_tilt_f_err, pst_tilt_f_err):
    pre_tilt_f_nrs1, pre_tilt_f_nrs2 = pre_tilt_f
    pst_tilt_f_nrs1, pst_tilt_f_nrs2 = pst_tilt_f
    pre_tilt_f_err_nrs1, pre_tilt_f_err_nrs2 = pre_tilt_f_err
    pst_tilt_f_err_nrs1, pst_tilt_f_err_nrs2 = pst_tilt_f_err

    pre_tilt_f_norm_nrs1 = np.median(pre_tilt_f_nrs1[:, :135], axis=-1)[:, None]
    pre_tilt_f_norm_nrs2 = np.median(pre_tilt_f_nrs2[:, :135], axis=-1)[:, None]

    pre_tilt_f_nrs1 /= pre_tilt_f_norm_nrs1
    pre_tilt_f_err_nrs1 /= pre_tilt_f_norm_nrs1
    pst_tilt_f_nrs1 /= pre_tilt_f_norm_nrs1
    pst_tilt_f_err_nrs1 /= pre_tilt_f_norm_nrs1
    pre_tilt_f_nrs2 /= pre_tilt_f_norm_nrs2
    pre_tilt_f_err_nrs2 /= pre_tilt_f_norm_nrs2
    pst_tilt_f_nrs2 /= pre_tilt_f_norm_nrs2
    pst_tilt_f_err_nrs2 /= pre_tilt_f_norm_nrs2

    pre_tilt_f = [pre_tilt_f_nrs1, pre_tilt_f_nrs2]
    pre_tilt_f_err = [pre_tilt_f_err_nrs1, pre_tilt_f_err_nrs2]
    pst_tilt_f = [pst_tilt_f_nrs1, pst_tilt_f_nrs2]
    pst_tilt_f_err = [pst_tilt_f_err_nrs1, pst_tilt_f_err_nrs2]
    return pre_tilt_f, pst_tilt_f, pre_tilt_f_err, pst_tilt_f_err


def WASP39b_lightcurve(resolution=None):
    (
        time_nrs1,
        time_nrs2,
        wavelength_nrs1,
        wavelength_nrs2,
        flux_nrs1,
        flux_nrs2,
        flux_err_nrs1,
        flux_err_nrs2,
    ) = load_binning(resolution)

    time, pre_tilt_f, pst_tilt_f, pre_tilt_f_err, pst_tilt_f_err = remove_tilt_event(
        time_nrs1,
        time_nrs2,
        flux_nrs1,
        flux_nrs2,
        flux_err_nrs1,
        flux_err_nrs2,
    )

    pre_tilt_f, pst_tilt_f, pre_tilt_f_err, pst_tilt_f_err = normalize_by_pretilt(
        pre_tilt_f, pst_tilt_f, pre_tilt_f_err, pst_tilt_f_err
    )
    return (
        time,
        [wavelength_nrs1, wavelength_nrs2],
        pre_tilt_f,
        pst_tilt_f,
        pre_tilt_f_err,
        pst_tilt_f_err,
    )


if __name__ == "__main__":
    pass
