# Introduction
This is an industrial-load cost calculator and optimizer for solar, batteries, and gas generation.

It can simulate an industrial load at locations worldwide with combinations of solar, battery, and gas generation. The output is a Levelized Cost of Energy (LCOE) in $/MWh and a yearly financial model.
 
The code calculates the LCOE using the following steps:
1. It pulls weather data for the speciifed `(lat, long)`
2. It simulates the solar power from the weather data
3. It simulates the powerflow of the system between the solar, battery, generator, and datacenter.
4. It calculates the annual cashflows and the LCOE of the system.

## Presets and system optimization

The Streamlit app includes Low, Middle, High, and High Cost/Cheaper Storage
assumption buttons, plus separate Lower Cost Gas and Higher Cost Gas buttons.
The cheaper-storage preset uses $175/MWh, represented internally as
$0.175/kWh because the BESS CAPEX input is denominated per kWh.

Four optimization buttons search for a low-LCOE design meeting a 95% or 99%
renewable-energy target with a gas turbine sized equal to the industrial load.
The two standalone options then keep that optimized Solar PV/BESS design,
remove the turbine and its costs, and calculate LCOE using only energy actually
served; unmet load has no outage penalty in the model. The optimizer screens a
load-scaled capacity grid and refines promising designs in 100 MW increments.
The final design is rerun through the original full power-flow and financial
models. Solar PV and BESS input limits scale with load to accommodate those
optimized designs.

# Usage
There are three ways to use this code:

## 1. Streamlit interface
`streamlit run app.py`

This fork keeps the original PVGIS-based solar, power-flow, and LCOE
methodology while adding a few workflow improvements:

* Calculations run only after pressing **Run calculation**. Changing an input
  does not trigger a new weather fetch or simulation; after a completed run,
  the app labels results as stale until the button is pressed again.
* Latitude and longitude are entered directly. The location preview uses
  Plotly's built-in political boundaries as a static visual guide; it does not
  use map tiles or allow the map to select coordinates.
* The power-flow chart shows the consecutive seven-day period with the lowest
  solar generation and exposes Plotly pan, zoom, reset, and scroll interactions.
* Capital Structure, CAPEX Costs, and O&M Rates are expanded by default.
* The default investment tax credit, combined tax rate, and soft-cost taxes
  are all 0%.


## 2. Command line interface
#### One-shot LCOE calculation
This simulates a single case.
```bash
python calculate_lcoe_one_shot.py --lat 31.9 --long -106.2 --solar-mw 250 --bess-mw 100 --generator-mw 125 --datacenter-load-mw 100
```

(See `calculate_lcoe_one_shot.py` for all possible args)

#### LCOE Ensemble Calculation
This simulates a range of cases and saves the results to a CSV file.
The "raw results" for every case are saved as a CSV, as well as the Pareto-optimal frontier on LCOE vs renewable-percentage. 
```bash
python run_ensemble.py
```
You can define the test cases in `run_ensemble.py`.

## 3. Python
```python
"""There are three steps to calculate the LCOE:
1. Get solar weather data
2. Simulate powerflow
3. Calculate LCOE
"""

# 1. Get solar weather data
solar_ac_dataframe = get_solar_ac_dataframe(lat, long)

# 2. Simulate powerflow
powerflow_results = simulate_system(lat, long, solar_ac_dataframe, ...)

# 3. Create DataCenter instance and calculate LCOE
datacenter = DataCenter(
    powerflow_results=powerflow_results,
    solar=100, 
    bess=100, 
    generator=125, 
    generator_type="Gas Engine",
    # CAPEX rates
    solar_capex_total_dollar_per_w=0.25,
    bess_capex_total_dollar_per_kwh=0.10,
    # O&M rates
    solar_om_fixed_dollar_per_kw=0.01,
    bess_om_fixed_dollar_per_kw=0.01,
    ... # See `datacenter.py` for all options and defaults
)

lcoe = datacenter.calculate_lcoe()
```

## Authors

* [Ben James](https://github.com/bengineer19)
