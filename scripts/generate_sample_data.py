import json
import sys

import numpy
import pandas

sys.path.append("src")
import atten_temp_functions

att_df = pandas.read_csv("data/FullDataSet_Randomized.txt")
bh_df = pandas.read_csv("data/waisdivide_imp.csv")

mean_acid = bh_df["acid [mol/L]"].mean()
mean_sscl = bh_df["sscl [mol/L]"].mean()

valid = att_df.dropna(subset=["atten_rate_C0"])
valid = valid[valid["atten_rate_C0"] > 0].copy()
quantiles = numpy.linspace(0.01, 0.99, 10)
targets = numpy.quantile(valid["atten_rate_C0"], quantiles)

indices = []
for t in targets:
    idx = (valid["atten_rate_C0"] - t).abs().idxmin()
    indices.append(idx)

sample = valid.loc[indices].reset_index(drop=True)

eV = 1.602176634e-19
func_kwargs = {
    "sigma0": 6.6e-6,
    "Epure": 0.55 * eV,
    "E_Hp": 0.20 * eV,
    "E_ssCl": 0.19 * eV,
    "E_cond": 0.22 * eV,
    "mu_Hp": 3.2,
    "mu_ssCl": 0.43,
}

atten_values = sample["atten_rate_C0"].to_numpy()

T_pure, terms_pure = atten_temp_functions.attenRateToTemperature(
    atten_values, return_terms=True, **func_kwargs
)

chem_imp = {"molar_Hp": mean_acid, "molar_ssCl": mean_sscl}
T_chem, terms_chem = atten_temp_functions.attenRateToTemperature(
    atten_values, chem_imp=chem_imp, return_terms=True, **func_kwargs
)

result = {
    "borehole": {
        "mean_acid_mol_per_L": mean_acid,
        "mean_sscl_mol_per_L": mean_sscl,
    },
    "points": [],
}

for i in range(10):
    point = {
        "x": sample.iloc[i]["x"],
        "y": sample.iloc[i]["y"],
        "attenuation_rate": sample.iloc[i]["atten_rate_C0"],
        "pure": {
            "temperature_K": float(T_pure[i]),
            "temperature_C": float(T_pure[i]) - 273.15,
        },
        "chem": {
            "temperature_K": float(T_chem[i]),
            "temperature_C": float(T_chem[i]) - 273.15,
        },
    }
    result["points"].append(point)

output_path = "outputs/sample_data.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")

print(f"Wrote {output_path}")
