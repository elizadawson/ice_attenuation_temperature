# Borehole Chemistry and Radar Attenuation to Temperature

This project estimates depth-averaged ice temperature from radar attenuation by converting attenuation rate to high-frequency electrical conductivity, $\sigma_{\infty}$, then inverting an Arrhenius-style conductivity model for temperature.

## Workflow

1. Load radar attenuation rates from a CSV
2. Load borehole chemistry or conductivity from a CSV (e.g. `waisdivide_imp.csv`)
3. Run `attenRateToTemperature` to estimate depth-averaged temperature from attenuation

## Project Structure

```
ice_attenuation_temperature/
├── data/                  # Input CSV files (not included in the repo)
├── notebooks/
│   └── englacial_temperature.ipynb  # Main notebook for running the inversion
├── outputs/               # Output CSV files with estimated temperatures
└── src/
    └── atten_temp_functions.py      # Core inversion functions
```
