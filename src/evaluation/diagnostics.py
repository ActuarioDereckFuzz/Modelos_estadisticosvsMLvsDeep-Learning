def glm_dispersion(model):
    pearson_dispersion = (
        model.pearson_chi2
        / model.df_resid
    )

    deviance_dispersion = (
        model.deviance
        / model.df_resid
    )

    return {
        "pearson_dispersion":
            pearson_dispersion,
        "deviance_dispersion":
            deviance_dispersion
    }