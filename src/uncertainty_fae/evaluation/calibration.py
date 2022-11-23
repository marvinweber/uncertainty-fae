QUANTILE_SIGMA_ENV_SCALES = {
    10: 0.13,
    20: 0.25,
    30: 0.39,
    40: 0.52,
    50: 0.67,
    60: 0.84,
    70: 1.04,
    80: 1.28,
    90: 1.65,
}


def observation_share_per_prediction_interval(
    mean: float,
    sigma: float,
    observations: list[float]
) -> dict:
    quantiles = {}
    observation_amount = len(observations)
    for quantile, sigma_env_scale in QUANTILE_SIGMA_ENV_SCALES.items():
        lower_bound = mean - (sigma_env_scale * sigma)
        upper_bound = mean + (sigma_env_scale * sigma)
        
        amount_in_interval = len([o for o in observations if o >= lower_bound and o <= upper_bound])
        quantiles[quantile] = amount_in_interval / observation_amount
    
    return quantiles
