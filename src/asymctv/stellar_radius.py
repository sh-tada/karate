import numpy as np
import jax.numpy as jnp
from jax import random

# PPL
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist


def robust_linear_regression(x, y):
    """
    Perform robust linear regression using Bayesian inference with
    a Student's t-distribution.

    This function normalizes the dependent variable `y` by its mean
    before performing the regression. The normalized `y` is used to
    make the model more numerically stable. Bayesian linear regression
    is performed using the NUTS (No-U-Turn Sampler) algorithm for MCMC
    sampling. The regression assumes a Student's t-distribution to
    improve robustness against outliers.

    Parameters
    ----------
    x : array-like
        Independent variable values (predictor).
    y : array-like
        Dependent variable values (response), which will be normalized
        by its mean.

    Returns
    -------
    intercept_samples : array-like
        Posterior samples for the intercept term of the regression,
        rescaled by the mean of `y`.
    slope_samples : array-like
        Posterior samples for the slope term of the regression, rescaled
        by the mean of `y`.
    samples : dict
        Dictionary containing all MCMC samples, including sigma and nu
        (t-distribution parameters).
    """

    y_mean = jnp.mean(y)
    y = y / y_mean
    x = jnp.array(x)
    y = jnp.array(y)

    def model(x, y=None):
        intercept = numpyro.sample("intercept", dist.Uniform(0, 10))
        slope = numpyro.sample("slope", dist.Uniform(0, 10.0))
        sigma = numpyro.sample("sigma", dist.Exponential(1))
        nu = numpyro.sample("nu", dist.Exponential(1))
        mu = intercept + slope * x
        numpyro.sample(
            "obs",
            dist.StudentT(df=nu, loc=mu[None, :] * jnp.ones_like(y), scale=sigma),
            obs=y,
        )

    # MCMC
    rng_key = random.PRNGKey(0)
    # initial_values = {
    #     "intercept": 1.00006,
    #     "slope": 0.0024,
    #     "sigma": 0.00283,
    #     "nu": 7.2,
    # }
    # init_strategy = init_to_value(values=initial_values)
    # nuts_kernel = NUTS(model, init_strategy=init_strategy)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000)
    mcmc.run(
        rng_key,
        x=x,
        y=y,
    )
    mcmc.print_summary()

    samples = mcmc.get_samples()
    intercept_samples = samples["intercept"]
    slope_samples = samples["slope"]
    return intercept_samples * y_mean, slope_samples * y_mean, samples


def rs_alpha_from_Tc_trend(
    Tc_intercept,
    Tc_slope,
    period,
    a_over_rs_b,
    cosi_b,
    wavelength,
    lambda_0,
):
    """
    Calculate the stellar radius at a given wavelength relative to
    the reference wavelength.

    This function computes the wavelength-dependent stellar radius
    (`rs_alpha`) based on the linear trend of the transit duration
    for the center of the planet's shadow disk (`Tc`). The trend is
    characterized by the intercept and slope.

    Parameters
    ----------
    Tc_intercept : float
        Intercept of the linear trend for Tc.
    Tc_slope : float
        Slope of the linear trend for the ratio of Tc.
    period : float
        Orbital period of the planet.
    a_over_rs_b : float
        Ratio of semi-major axis to stellar radius.
    cosi_b : float
        Cosine of the orbital inclination angle.
    wavelength : float
        Wavelength at which to calculate the stellar radius.
    lambda_0 : float
        Reference wavelength for normalization.

    Returns
    -------
    rs_alpha : float
        Ratio of the stellar radius at the given wavelength to the
        stellar radius at the reference wavelength.
    """

    Tc_lambda = Tc_intercept + Tc_slope * (wavelength - lambda_0)

    # Conversion from the duration to Rs
    theta_lambda = 2 * np.pi * Tc_lambda / period / 2
    rs_lambda = np.sqrt(
        a_over_rs_b**2
        * (np.sin(theta_lambda) ** 2 + np.cos(theta_lambda) ** 2 * cosi_b**2)
    )

    # Conversion from the duration to Rs at lambda_0
    theta_lambda0 = 2 * np.pi * Tc_intercept / period / 2
    rs_lambda0 = np.sqrt(
        a_over_rs_b**2
        * (np.sin(theta_lambda0) ** 2 + np.cos(theta_lambda0) ** 2 * cosi_b**2)
    )
    rs_alpha = rs_lambda / rs_lambda0
    return rs_alpha


if __name__ == "__main__":
    pass
