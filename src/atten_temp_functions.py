import numpy as np
from scipy.optimize import root_scalar, fsolve

def attenRateToTemperature(
    attenu_rate,
    chem_imp=None,
    cond=None,
    return_terms=False,
    Tref=251.0,
    **kwargs
):
    if chem_imp is not None and cond is not None:
        raise ValueError("Specify only one of chem_imp or cond, not both.")

    if np.isscalar(attenu_rate) and np.isnan(attenu_rate):
        return np.nan

    k = 1.380649e-23
    c = 3e8
    eps0 = 8.854e-12
    epsr = 3.17
    eV = 1.602176634e-19

    attenu_rate = np.atleast_1d(attenu_rate)

    sigma = attenu_rate * c * eps0 * np.sqrt(epsr) / (
        1000 * (10 * np.log10(np.exp(1)))
    )

    sigma0 = kwargs.get("sigma0", 6.6e-6)
    Epure = kwargs.get("Epure", 0.55 * eV)
    E_Hp = kwargs.get("E_Hp", 0.20 * eV)
    E_ssCl = kwargs.get("E_ssCl", 0.19 * eV)
    E_cond = kwargs.get("E_cond", 0.22 * eV)
    mu_Hp = kwargs.get("mu_Hp", 3.2)
    mu_ssCl = kwargs.get("mu_ssCl", 0.43)
    molar_ssCl = kwargs.get("molar_ssCl", 4.2e-6)
    molar_Hp = kwargs.get("molar_Hp", 2.7e-6)

    if chem_imp is not None:
        molar_Hp = chem_imp.get("molar_Hp", molar_Hp)
        molar_ssCl = chem_imp.get("molar_ssCl", molar_ssCl)

    def sigma_residual(T_val, sigma_obs):
        term_pure = sigma0 * np.exp((Epure / k) * (1 / Tref - 1 / T_val))

        if cond is not None:
            term_cond = cond * np.exp((E_cond / k) * (1 / Tref - 1 / T_val))
            sigma_model = term_cond

        elif chem_imp is not None:
            term_Hp = mu_Hp * molar_Hp * np.exp((E_Hp / k) * (1 / Tref - 1 / T_val))
            term_ssCl = mu_ssCl * molar_ssCl * np.exp((E_ssCl / k) * (1 / Tref - 1 / T_val))
            sigma_model = term_pure + term_Hp + term_ssCl

        else:
            sigma_model = term_pure

        return sigma_model - sigma_obs

    T = np.full_like(sigma, np.nan, dtype=float)

    for ii in range(sigma.size):
        sigma_val = sigma.flat[ii]

        if not np.isfinite(sigma_val) or sigma_val <= 0:
            continue

        try:
            sol = root_scalar(
                sigma_residual,
                args=(sigma_val,),
                bracket=[150, 350],
                method="brentq",
            )
            T.flat[ii] = sol.root
        except ValueError:
            T.flat[ii] = fsolve(sigma_residual, 250, args=(sigma_val,))[0]

    if not return_terms:
        return T if T.size > 1 else T.item()

    term_pure_out = sigma0 * np.exp((Epure / k) * (1 / Tref - 1 / T))

    if cond is not None:
        term_cond_out = cond * np.exp((E_cond / k) * (1 / Tref - 1 / T))
        term_Hp_out = np.zeros_like(T)
        term_ssCl_out = np.zeros_like(T)
        sigma_model_out = term_cond_out
        term_pure_report = np.zeros_like(T)

    elif chem_imp is not None:
        term_Hp_out = mu_Hp * molar_Hp * np.exp((E_Hp / k) * (1 / Tref - 1 / T))
        term_ssCl_out = mu_ssCl * molar_ssCl * np.exp((E_ssCl / k) * (1 / Tref - 1 / T))
        term_cond_out = np.zeros_like(T)
        sigma_model_out = term_pure_out + term_Hp_out + term_ssCl_out
        term_pure_report = term_pure_out

    else:
        term_Hp_out = np.zeros_like(T)
        term_ssCl_out = np.zeros_like(T)
        term_cond_out = np.zeros_like(T)
        sigma_model_out = term_pure_out
        term_pure_report = term_pure_out

    terms = {
        "sigma_pure": term_pure_report,
        "sigma_Hp": term_Hp_out,
        "sigma_ssCl": term_ssCl_out,
        "sigma_cond": term_cond_out,
        "sigma_model": sigma_model_out,
        "sigma_obs": sigma,
    }

    if T.size == 1:
        T = T.item()
        terms = {
            k: v.item() if np.ndim(v) == 1 and len(np.atleast_1d(v)) == 1 else v
            for k, v in terms.items()
        }

    return T, terms
