"""Module for handling user inputs in the Streamlit app."""

import math
import streamlit as st
import pandas as pd
from typing import Dict
import plotly.graph_objects as go

from app_components.st_outputs import create_capacity_chart
from core.defaults import (
    BESS_HRS_STORAGE, DATACENTER_DEMAND_MW, DEFAULTS_GENERATORS, DEFAULTS_SOLAR_CAPEX, DEFAULTS_BESS_CAPEX,
    DEFAULTS_SYSTEM_INTEGRATION_CAPEX, DEFAULTS_SOFT_COSTS_CAPEX, DEFAULTS_OM, DEFAULTS_FINANCIAL,
    DEFAULTS_DEPRECIATION_SCHEDULE
)
import reverse_geocoder as rg

# Amarillo, TX
MAP_INITIAL_LAT = 35.199
MAP_INITIAL_LONG = -101.845


_SOFT_COST_WIDGET_KEYS = (
    "soft_general", "soft_epc", "soft_design", "soft_permit",
    "soft_startup", "soft_insurance", "soft_taxes",
)
_COST_PRESET_VALUES = {
    "Low Cost": {
        "debt_cost": 8.0, "leverage": 100.0, "debt_term": 20,
        "equity_cost": 0.0, "itc": 0.0, "tax_rate": 0.0,
        "pv_modules": 0.530, "pv_inverters": 0.0, "pv_racking": 0.0,
        "pv_bos": 0.0, "pv_labor": 0.0,
        "bess_units": 62.0, "bess_bos": 0, "bess_labor": 0,
        "si_microgrid": 100, "si_controls": 0, "si_labor": 0,
        "solar_om": 4, "bess_om": 0.8, "bos_om": 2.0,
        "soft_om": 0.0, "om_escalator": 0.0, "fuel_escalator": 0.0,
        "soft_general": 0.0, "soft_epc": 0.0, "soft_design": 0.0,
        "soft_permit": 0.0, "soft_startup": 0.0,
        "soft_insurance": 0.0, "soft_taxes": 0.0,
    },
    "Middle Cost": {
        "debt_cost": 9.0, "leverage": 100.0, "debt_term": 20,
        "equity_cost": 0.0, "itc": 0.0, "tax_rate": 0.0,
        "pv_modules": 0.667, "pv_inverters": 0.0, "pv_racking": 0.0,
        "pv_bos": 0.0, "pv_labor": 0.0,
        "bess_units": 140.0, "bess_bos": 0, "bess_labor": 0,
        "si_microgrid": 200, "si_controls": 0, "si_labor": 0,
        "solar_om": 7, "bess_om": 1.7, "bos_om": 4.0,
        "soft_om": 0.0, "om_escalator": 0.0, "fuel_escalator": 0.0,
        "soft_general": 0.0, "soft_epc": 0.0, "soft_design": 0.0,
        "soft_permit": 0.0, "soft_startup": 0.0,
        "soft_insurance": 0.0, "soft_taxes": 0.0,
    },
    "High Cost": {
        "debt_cost": 10.0, "leverage": 100.0, "debt_term": 20,
        "equity_cost": 0.0, "itc": 0.0, "tax_rate": 0.0,
        "pv_modules": 0.785, "pv_inverters": 0.0, "pv_racking": 0.0,
        "pv_bos": 0.0, "pv_labor": 0.0,
        "bess_units": 378.0, "bess_bos": 0, "bess_labor": 0,
        "si_microgrid": 300, "si_controls": 0, "si_labor": 0,
        "solar_om": 11, "bess_om": 2.5, "bos_om": 6.0,
        "soft_om": 0.0, "om_escalator": 0.0, "fuel_escalator": 0.0,
        "soft_general": 0.0, "soft_epc": 0.0, "soft_design": 0.0,
        "soft_permit": 0.0, "soft_startup": 0.0,
        "soft_insurance": 0.0, "soft_taxes": 0.0,
    },
    # The app stores BESS CAPEX in $/kWh, so $175/MWh is $0.175/kWh.
    "High Cost, Cheaper Storage": {
        "debt_cost": 10.0, "leverage": 100.0, "debt_term": 20,
        "equity_cost": 0.0, "itc": 0.0, "tax_rate": 0.0,
        "pv_modules": 0.785, "pv_inverters": 0.0, "pv_racking": 0.0,
        "pv_bos": 0.0, "pv_labor": 0.0,
        "bess_units": 0.175, "bess_bos": 0, "bess_labor": 0,
        "si_microgrid": 300, "si_controls": 0, "si_labor": 0,
        "solar_om": 11, "bess_om": 2.5, "bos_om": 6.0,
        "soft_om": 0.0, "om_escalator": 0.0, "fuel_escalator": 0.0,
        "soft_general": 0.0, "soft_epc": 0.0, "soft_design": 0.0,
        "soft_permit": 0.0, "soft_startup": 0.0,
        "soft_insurance": 0.0, "soft_taxes": 0.0,
    },
}

_GAS_PRESETS = {
    "Lower Cost Gas (default)": {
        "gensets": 635, "gen_bos": 0, "gen_labor": 0,
        "gen_om_fixed": 15.0, "fuel_price": 8.0,
    },
    "Higher Cost Gas": {
        "gensets": 1_000, "gen_bos": 0, "gen_labor": 0,
        "gen_om_fixed": 32.0, "fuel_price": 11.0,
    },
}
_COST_PRESET_KEYS = set(_COST_PRESET_VALUES["High Cost"])
_GAS_PRESET_KEYS = set(_GAS_PRESETS["Higher Cost Gas"]) | {"gen_type"}


def _apply_widget_values(values: Dict, label_key: str, label: str) -> None:
    """Apply a preset before widgets are created on the next Streamlit rerun."""
    for key, value in values.items():
        st.session_state[key] = value
        if key != "depreciation_schedule":
            st.query_params[key] = value
    st.session_state[label_key] = label


def apply_cost_preset(preset_name: str) -> None:
    values = dict(_COST_PRESET_VALUES[preset_name])
    values["depreciation_schedule"] = pd.DataFrame(
        {"Year": range(1, 21), "Depreciation (%)": [0.0] * 20}
    )
    _apply_widget_values(values, "active_cost_preset", preset_name)


def apply_gas_preset(preset_name: str) -> None:
    values = dict(_GAS_PRESETS[preset_name])
    values["gen_type"] = "Gas Turbine"
    _apply_widget_values(values, "active_gas_preset", preset_name)


def create_preset_controls() -> None:
    """Render explicit cost and gas assumption preset buttons."""
    st.subheader("Quick assumption presets")
    st.caption(
        "Cost presets replace the financial, CAPEX, and non-gas O&M inputs. "
        "Gas cost is selected separately; you can edit any value afterward."
    )
    cost_columns = st.columns(4)
    for column, (label, preset_key) in zip(
        cost_columns,
        [
            ("Low Cost", "Low Cost"),
            ("Middle Cost", "Middle Cost"),
            ("High Cost", "High Cost"),
            ("High Cost, Cheaper Storage", "High Cost, Cheaper Storage"),
        ],
    ):
        with column:
            st.button(
                label,
                key=f"apply_cost_{preset_key.lower().replace(' ', '_').replace(',', '')}",
                help=(
                    "Uses high-cost assumptions with BESS CAPEX at $175/MWh "
                    "($0.175/kWh)."
                    if preset_key == "High Cost, Cheaper Storage"
                    else f"Apply the {preset_key.lower()} cost assumptions."
                ),
                on_click=apply_cost_preset,
                args=(preset_key,),
                use_container_width=True,
            )

    if "active_cost_preset" not in st.session_state:
        st.session_state.active_cost_preset = "Current input values"
    st.caption(f"Cost preset: {st.session_state.active_cost_preset}")

    gas_columns = st.columns(2)
    for column, preset_name in zip(gas_columns, _GAS_PRESETS):
        with column:
            st.button(
                preset_name,
                key=f"apply_gas_{preset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                help=(
                    "$635/kW gas-turbine CAPEX, $15/kW-year fixed O&M, "
                    "$8/MMBtu fuel. This is the default."
                    if preset_name.startswith("Lower")
                    else "$1,000/kW gas-turbine CAPEX, $32/kW-year fixed O&M, "
                    "$11/MMBtu fuel."
                ),
                on_click=apply_gas_preset,
                args=(preset_name,),
                use_container_width=True,
            )
    if "active_gas_preset" not in st.session_state:
        if any(
            key in st.query_params
            for key in ("gensets", "gen_om_fixed", "fuel_price")
        ):
            st.session_state.active_gas_preset = "Custom (URL values)"
        else:
            apply_gas_preset("Lower Cost Gas (default)")
    st.caption(f"Gas preset: {st.session_state.active_gas_preset}")


def calculate_capex_subtotals(inputs: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate CAPEX subtotals for each system component.
    
    Returns:
        Dict with both unit rates and absolute totals for each component:
        {
            'solar': {'rate': $/W, 'absolute': $M},
            'bess': {'rate': $/kWh, 'absolute': $M},
            'generator': {'rate': $/kW, 'absolute': $M},
            'system_integration': {'rate': $/kW, 'absolute': $M},
            'soft_costs': {'rate': %, 'absolute': $M}
        }
    """
    solar_capacity_w = inputs['solar_pv_capacity_mw'] * 1_000_000
    # Calculate Solar unit rate and absolute CAPEX
    solar_rate = (
        inputs['capex_pv_modules'] + 
        inputs['capex_pv_inverters'] + 
        inputs['capex_pv_racking'] + 
        inputs['capex_pv_balance_system'] + 
        inputs['capex_pv_labor']
    )
    solar_absolute = solar_capacity_w * solar_rate
    
    # Calculate BESS unit rate and absolute CAPEX
    bess_rate = (
        inputs['capex_bess_units'] + 
        inputs['capex_bess_balance_of_system'] + 
        inputs['capex_bess_labor']
    )
    bess_system_mwh = inputs['bess_max_power_mw'] * BESS_HRS_STORAGE
    bess_absolute = bess_system_mwh * 1000 * bess_rate
    
    # Calculate Generator unit rate and absolute CAPEX
    generator_rate = (
        inputs['capex_gensets'] + 
        inputs['capex_gen_balance_of_system'] + 
        inputs['capex_gen_labor']
    )
    generator_absolute = inputs['generator_capacity_mw'] * 1000 * generator_rate
    
    # Calculate System Integration unit rate and absolute CAPEX
    system_integration_rate = (
        inputs['capex_si_microgrid'] + 
        inputs['capex_si_controls'] + 
        inputs['capex_si_labor']
    )
    system_integration_absolute = inputs['datacenter_load_mw'] * 1000 * system_integration_rate
    
    # Calculate total hard costs
    total_hard_costs = (
        solar_absolute +
        bess_absolute +
        generator_absolute +
        system_integration_absolute
    )
    
    # Calculate soft costs rate and absolute
    soft_costs_rate = (
        inputs['capex_soft_costs_general_conditions'] +
        inputs['capex_soft_costs_epc_overhead'] +
        inputs['capex_soft_costs_design_engineering'] +
        inputs['capex_soft_costs_permitting'] +
        inputs['capex_soft_costs_startup'] +
        inputs['capex_soft_costs_insurance'] +
        inputs['capex_soft_costs_taxes']
    )
    soft_costs_absolute = total_hard_costs * soft_costs_rate / 100
    
    # Return both rates and absolute values (absolute in millions)
    return {
        'solar': {
            'rate': solar_rate,
            'total_absolute': solar_absolute / 1_000_000,
            'components_absolute': {
                'pv_modules': solar_capacity_w * inputs['capex_pv_modules'],
                'pv_inverters': solar_capacity_w * inputs['capex_pv_inverters'],
                'pv_racking': solar_capacity_w * inputs['capex_pv_racking'],
                'pv_balance_system': solar_capacity_w * inputs['capex_pv_balance_system'],
                'pv_labor': solar_capacity_w * inputs['capex_pv_labor']
            }
        },
        'bess': {
            'rate': bess_rate,
            'total_absolute': bess_absolute / 1_000_000,
            'components_absolute': {
                'bess_units': bess_system_mwh * 1000 * inputs['capex_bess_units'],
                'bess_balance_of_system': bess_system_mwh * 1000 * inputs['capex_bess_balance_of_system'],
                'bess_labor': bess_system_mwh * 1000 * inputs['capex_bess_labor']
            }
        },
        'generator': {
            'rate': generator_rate,
            'total_absolute': generator_absolute / 1_000_000,
            'components_absolute': {
                'gensets': inputs['generator_capacity_mw'] * 1000 * inputs['capex_gensets'],
                'gen_balance_of_system': inputs['generator_capacity_mw'] * 1000 * inputs['capex_gen_balance_of_system'],
                'gen_labor': inputs['generator_capacity_mw'] * 1000 * inputs['capex_gen_labor']
            }
        },
        'system_integration': {
            'rate': system_integration_rate,
            'total_absolute': system_integration_absolute / 1_000_000,
            'components_absolute': {
                'microgrid': inputs['datacenter_load_mw'] * 1000 * inputs['capex_si_microgrid'],
                'controls': inputs['datacenter_load_mw'] * 1000 * inputs['capex_si_controls'],
                'labor': inputs['datacenter_load_mw'] * 1000 * inputs['capex_si_labor']
            }
        },
        'soft_costs': {
            'rate': soft_costs_rate,
            'total_absolute': soft_costs_absolute / 1_000_000,
            'components_absolute': {
                'general_conditions': total_hard_costs * inputs['capex_soft_costs_general_conditions'] / 100,
                'epc_overhead': total_hard_costs * inputs['capex_soft_costs_epc_overhead'] / 100,
                'design_engineering': total_hard_costs * inputs['capex_soft_costs_design_engineering'] / 100,
                'permitting': total_hard_costs * inputs['capex_soft_costs_permitting'] / 100,
                'startup': total_hard_costs * inputs['capex_soft_costs_startup'] / 100,
                'insurance': total_hard_costs * inputs['capex_soft_costs_insurance'] / 100,
                'taxes': total_hard_costs * inputs['capex_soft_costs_taxes'] / 100
            }
        }
    }

def create_system_inputs() -> Dict:
    """Create all input sections in the Streamlit app."""
    st.subheader("System Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    # Get query parameters
    query_params = st.query_params
    
    def update_param(key: str):
        st.query_params[key] = st.session_state[key]
        if key in _GAS_PRESET_KEYS:
            st.session_state.active_gas_preset = "Custom (edited)"
    
    with col1:
        datacenter_load = st.number_input(
            "Industrial Load Demand (MW)",
            value=int(query_params.get("dc_load", DATACENTER_DEMAND_MW)),
            min_value=0,
            max_value=1002,
            step=50,
            key="dc_load",
            on_change=update_param,
            args=("dc_load",)
        )

    # Extend the component inputs with load so optimized results can be
    # represented directly in the controls, including locations with seasonal
    # storage requirements above the former 10 GW cap.
    solar_capacity_limit = max(
        10_000,
        int(math.ceil(datacenter_load * 150 / 100) * 100),
        int(st.session_state.get("solar", query_params.get("solar", 0)) or 0),
    )
    bess_capacity_limit = max(
        10_000,
        int(math.ceil(datacenter_load * 500 / 100) * 100),
        int(st.session_state.get("bess", query_params.get("bess", 0)) or 0),
    )
            
    with col2:
        solar_pv_capacity = st.number_input(
            "Solar PV Capacity (MW DC)",
            value=min(int(query_params.get("solar", 2500)), solar_capacity_limit),
            min_value=0,
            max_value=solar_capacity_limit,
            step=50,
            key="solar",
            help=f"The input limit scales with load; current maximum is {solar_capacity_limit:,} MW.",
            on_change=update_param,
            args=("solar",)
        )
            
    with col3:
        bess_max_power = st.number_input(
            "BESS Power (MW), 4hr store",
            value=min(int(query_params.get("bess", 2500)), bess_capacity_limit),
            min_value=0,
            max_value=bess_capacity_limit,
            step=50,
            key="bess",
            help=f"The input limit scales with load; current maximum is {bess_capacity_limit:,} MW.",
            on_change=update_param,
            args=("bess",)
        )
            
    with col4:
        generator_capacity = st.number_input(
            "Generator Capacity (MW)",
            value=min(int(query_params.get("gen", 1000)), max(1000, int(datacenter_load))),
            min_value=0,
            max_value=max(
                1000,
                int(datacenter_load),
                int(st.session_state.get("gen", query_params.get("gen", 0)) or 0),
            ),
            step=10,
            key="gen",
            on_change=update_param,
            args=("gen",)
        )
            
        generator_type = st.selectbox(
            "Generator Type",
            ["Gas Engine", "Gas Turbine"],
            index=0 if query_params.get("gen_type", "Gas Turbine") == "Gas Engine" else 1,
            key="gen_type",
            on_change=update_param,
            args=("gen_type",)
        )
    
    # Display capacity chart
    st.plotly_chart(
        create_capacity_chart(datacenter_load, solar_pv_capacity, bess_max_power, generator_capacity),
        use_container_width=True
    )
    st.divider()

    return {
        'datacenter_load_mw': datacenter_load,
        'solar_pv_capacity_mw': solar_pv_capacity,
        'bess_max_power_mw': bess_max_power,
        'generator_capacity_mw': generator_capacity,
        'generator_type': generator_type,
    }

def create_map_input() -> Dict:
    """Collect coordinates and display a non-interactive visual location guide.

    Coordinates are entered explicitly.  The map is intentionally configured as
    a visual reference only: it has no coordinate-selection callback, uses
    Plotly's built-in political boundaries instead of map tiles, and is rendered
    as a static chart.
    """
    st.subheader("Location")
    st.write("Enter the coordinates to evaluate. The map is a visual guide only.")

    query_params = st.query_params

    def update_location_param(key: str):
        st.query_params[key] = st.session_state[key]

    lat_col, long_col = st.columns(2)
    with lat_col:
        latitude = st.number_input(
            "Latitude (°)",
            value=float(query_params.get("lat", MAP_INITIAL_LAT)),
            min_value=-90.0,
            max_value=90.0,
            step=0.01,
            format="%.5f",
            key="lat",
            on_change=update_location_param,
            args=("lat",)
        )
    with long_col:
        longitude = st.number_input(
            "Longitude (°)",
            value=float(query_params.get("long", MAP_INITIAL_LONG)),
            min_value=-180.0,
            max_value=180.0,
            step=0.01,
            format="%.5f",
            key="long",
            on_change=update_location_param,
            args=("long",)
        )

    lat_long_tuple = (latitude, longitude)
    rg_result = rg.search(lat_long_tuple, mode=1)[0]
    location_name = f"{rg_result['name']}, {rg_result['admin1']} ({rg_result['cc']})"
    st.caption(f"Selected location: {location_name}")

    map_context = st.radio(
        "Map context",
        ["Regional context", "Global context"],
        horizontal=True,
        key="location_map_context",
        label_visibility="collapsed",
    )
    regional_context = map_context == "Regional context"

    location_map = go.Figure()
    location_map.add_trace(
        go.Scattergeo(
            lon=[longitude],
            lat=[latitude],
            mode="markers+text",
            text=[location_name],
            textposition="top right",
            textfont={"size": 11, "color": "#1e2b25"},
            marker={
                "size": 11,
                "color": "#287b4f",
                "line": {"width": 2, "color": "#ffffff"},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    location_map.update_geos(
        scope="world",
        projection_type="natural earth",
        projection_scale=4.5 if regional_context else 1,
        center={"lat": latitude, "lon": longitude} if regional_context else None,
        showland=True,
        landcolor="#dce8df",
        showocean=True,
        oceancolor="#e8f1f4",
        showcountries=True,
        countrycolor="#95a89b",
        countrywidth=0.6,
        showcoastlines=True,
        coastlinecolor="#95a89b",
        coastlinewidth=0.6,
        showlakes=True,
        lakecolor="#e8f1f4",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    location_map.update_layout(
        height=370,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(
        location_map,
        use_container_width=True,
        config={"staticPlot": True, "displayModeBar": False, "scrollZoom": False},
        key="location_map",
    )

    return latitude, longitude, location_name


def create_financial_inputs(generator_type: str) -> Dict:
    st.subheader("Financial Inputs")
    
    # Get query parameters
    query_params = st.query_params
    
    def update_param(key: str):
        st.query_params[key] = st.session_state[key]
        if key in _COST_PRESET_KEYS:
            st.session_state.active_cost_preset = "Custom (edited)"
        if key in _GAS_PRESET_KEYS:
            st.session_state.active_gas_preset = "Custom (edited)"
    
    # Financial Inputs
    with st.expander("Capital Structure", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cost_of_debt = st.number_input(
                "Cost of Debt (%)",
                value=float(query_params.get("debt_cost", DEFAULTS_FINANCIAL['cost_of_debt_pct'])),
                min_value=0.0,
                max_value=100.0,
                key="debt_cost",
                on_change=update_param,
                args=("debt_cost",)
            )
            leverage = st.number_input(
                "Leverage (%)",
                value=float(query_params.get("leverage", DEFAULTS_FINANCIAL['leverage_pct'])),
                min_value=0.0,
                max_value=100.0,
                key="leverage",
                on_change=update_param,
                args=("leverage",)
            )
            debt_term = st.number_input(
                "Debt Term (years)",
                value=int(query_params.get("debt_term", DEFAULTS_FINANCIAL['debt_term_years'])),
                min_value=1,
                key="debt_term",
                on_change=update_param,
                args=("debt_term",)
            )
            cost_of_equity = st.number_input(
                "Cost of Equity (%)",
                value=float(query_params.get("equity_cost", DEFAULTS_FINANCIAL['cost_of_equity_pct'])),
                min_value=0.0,
                max_value=100.0,
                key="equity_cost",
                on_change=update_param,
                args=("equity_cost",)
            )
            investment_tax_credit_pct = st.number_input(
                "Investment Tax Credit (%)",
                value=float(query_params.get("itc", DEFAULTS_FINANCIAL['investment_tax_credit_pct'])),
                min_value=0.0,
                max_value=100.0,
                key="itc",
                on_change=update_param,
                args=("itc",)
            )
            combined_tax_rate = st.number_input(
                "Combined Tax Rate (%)",
                value=float(query_params.get("tax_rate", DEFAULTS_FINANCIAL['combined_tax_rate_pct'])),
                min_value=0.0,
                max_value=100.0,
                key="tax_rate",
                on_change=update_param,
                args=("tax_rate",)
            )
        
        with col2:
            # Create default MACRS depreciation schedule (20 years)
            if 'depreciation_schedule' not in st.session_state:
                st.session_state.depreciation_schedule = pd.DataFrame({
                    'Year': range(1, 21),
                    'Depreciation (%)': DEFAULTS_DEPRECIATION_SCHEDULE
                })
            
            # Display editable depreciation schedule
            edited_depreciation = st.data_editor(
                st.session_state.depreciation_schedule,
                column_config={
                    "Year": st.column_config.NumberColumn(
                        "Year",
                        help="Year of depreciation",
                        min_value=1,
                        max_value=20,
                        step=1,
                        disabled=True
                    ),
                    "Depreciation (%)": st.column_config.NumberColumn(
                        "Depreciation (%)",
                        help="Percentage of total CAPEX to depreciate in this year",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        format="%.1f%%"
                    )
                },
                hide_index=True,
                width=400
            )
            
            # Update session state with edited values
            st.session_state.depreciation_schedule = edited_depreciation
    
    # CAPEX Inputs
    with st.expander("CAPEX Costs", expanded=True):
        # Solar PV
        st.subheader("Solar PV")
        col1, col2 = st.columns(2)
        with col1:
            pv_modules = st.number_input(
                "Modules ($/W)",
                value=float(query_params.get("pv_modules", DEFAULTS_SOLAR_CAPEX['modules'])),
                format="%.3f",
                key="pv_modules",
                on_change=update_param,
                args=("pv_modules",)
            )
            pv_inverters = st.number_input(
                "Inverters ($/W)",
                value=float(query_params.get("pv_inverters", DEFAULTS_SOLAR_CAPEX['inverters'])),
                format="%.3f",
                key="pv_inverters",
                on_change=update_param,
                args=("pv_inverters",)
            )
            pv_racking = st.number_input(
                "Racking and Foundations ($/W)",
                value=float(query_params.get("pv_racking", DEFAULTS_SOLAR_CAPEX['racking'])),
                format="%.3f",
                key="pv_racking",
                on_change=update_param,
                args=("pv_racking",)
            )
        with col2:
            pv_balance_system = st.number_input(
                "Balance of System ($/W)",
                value=float(query_params.get("pv_bos", DEFAULTS_SOLAR_CAPEX['balance_of_system'])),
                format="%.3f",
                key="pv_bos",
                on_change=update_param,
                args=("pv_bos",)
            )
            pv_labor = st.number_input(
                "Labor ($/W)",
                value=float(query_params.get("pv_labor", DEFAULTS_SOLAR_CAPEX['labor'])),
                format="%.3f",
                key="pv_labor",
                on_change=update_param,
                args=("pv_labor",)
            )
    
        # BESS
        st.subheader("Battery Energy Storage System")
        col1, col2 = st.columns(2)
        with col1:
            bess_units = st.number_input(
                "BESS Units ($/kWh)",
                value=float(query_params.get("bess_units", DEFAULTS_BESS_CAPEX['units'])),
                min_value=0.0,
                step=0.1,
                format="%.3f",
                key="bess_units",
                on_change=update_param,
                args=("bess_units",)
            )
            bess_balance_of_system = st.number_input(
                "Balance of System ($/kWh)",
                value=int(query_params.get("bess_bos", DEFAULTS_BESS_CAPEX['balance_of_system'])),
                format="%d",
                key="bess_bos",
                on_change=update_param,
                args=("bess_bos",)
            )
        with col2:
            bess_labor = st.number_input(
                "Labor ($/kWh)",
                value=int(query_params.get("bess_labor", DEFAULTS_BESS_CAPEX['labor'])),
                format="%d",
                key="bess_labor",
                on_change=update_param,
                args=("bess_labor",)
            )

        # Generators
        st.subheader("Generators")
        col1, col2 = st.columns(2)
        gen_config = DEFAULTS_GENERATORS[generator_type]
        with col1:
            gensets = st.number_input(
                "Gensets ($/kW)", 
                value=int(query_params.get("gensets", gen_config['capex']['gensets'])),
                format="%d",
                key="gensets",
                on_change=update_param,
                args=("gensets",)
            )
            gen_balance_of_system = st.number_input(
                "Balance of System ($/kW)", 
                value=int(query_params.get("gen_bos", gen_config['capex']['balance_of_system'])),
                format="%d",
                key="gen_bos",
                on_change=update_param,
                args=("gen_bos",)
            )
        with col2:
            gen_labor = st.number_input(
                "Labor ($/kW)", 
                value=int(query_params.get("gen_labor", gen_config['capex']['labor'])),
                format="%d",
                key="gen_labor",
                on_change=update_param,
                args=("gen_labor",)
            )

        # System Integration
        st.subheader("System Integration")
        col1, col2 = st.columns(2)
        with col1:
            si_microgrid = st.number_input(
                "Microgrid Switchgear, Transformers, etc. ($/kW)",
                value=int(query_params.get("si_microgrid", DEFAULTS_SYSTEM_INTEGRATION_CAPEX['microgrid'])),
                format="%d",
                key="si_microgrid",
                on_change=update_param,
                args=("si_microgrid",)
            )
            si_controls = st.number_input(
                "Controls ($/kW)",
                value=int(query_params.get("si_controls", DEFAULTS_SYSTEM_INTEGRATION_CAPEX['controls'])),
                format="%d",
                key="si_controls",
                on_change=update_param,
                args=("si_controls",)
            )
        with col2:
            si_labor = st.number_input(
                "System Integration Labor ($/kW)",
                value=int(query_params.get("si_labor", DEFAULTS_SYSTEM_INTEGRATION_CAPEX['labor'])),
                format="%d",
                key="si_labor",
                on_change=update_param,
                args=("si_labor",)
            )

        # Soft Costs
        st.subheader("Soft Costs (CAPEX)")
        col1, col2 = st.columns(2)
        with col1:
            soft_costs_general_conditions = st.number_input(
                "General Conditions (%)",
                value=float(query_params.get("soft_general", DEFAULTS_SOFT_COSTS_CAPEX['general_conditions'])),
                format="%.2f",
                key="soft_general",
                on_change=update_param,
                args=("soft_general",)
            )
            soft_costs_epc_overhead = st.number_input(
                "EPC Overhead (%)",
                value=float(query_params.get("soft_epc", DEFAULTS_SOFT_COSTS_CAPEX['epc_overhead'])),
                format="%.2f",
                key="soft_epc",
                on_change=update_param,
                args=("soft_epc",)
            )
            soft_costs_design_engineering = st.number_input(
                "Design, Engineering, and Surveys (%)",
                value=float(query_params.get("soft_design", DEFAULTS_SOFT_COSTS_CAPEX['design_engineering'])),
                format="%.2f",
                key="soft_design",
                on_change=update_param,
                args=("soft_design",)
            )
        with col2:
            soft_costs_permitting = st.number_input(
                "Permitting & Inspection (%)",
                value=float(query_params.get("soft_permit", DEFAULTS_SOFT_COSTS_CAPEX['permitting'])),
                format="%.2f",
                key="soft_permit",
                on_change=update_param,
                args=("soft_permit",)
            )
            soft_costs_startup = st.number_input(
                "Startup & Commissioning (%)",
                value=float(query_params.get("soft_startup", DEFAULTS_SOFT_COSTS_CAPEX['startup'])),
                format="%.2f",
                key="soft_startup",
                on_change=update_param,
                args=("soft_startup",)
            )
            soft_costs_insurance = st.number_input(
                "Insurance (%)",
                value=float(query_params.get("soft_insurance", DEFAULTS_SOFT_COSTS_CAPEX['insurance'])),
                format="%.2f",
                key="soft_insurance",
                on_change=update_param,
                args=("soft_insurance",)
            )
            soft_costs_taxes = st.number_input(
                "Taxes (%)",
                value=float(query_params.get("soft_taxes", DEFAULTS_SOFT_COSTS_CAPEX['taxes'])),
                format="%.2f",
                key="soft_taxes",
                on_change=update_param,
                args=("soft_taxes",)
            )
    
    # O&M Inputs
    with st.expander("O&M Rates", expanded=True):
        col1, col2 = st.columns(2)
        
        # Column 1: Asset-specific O&M
        with col1:
            st.subheader("Operations and Maintenance")
            fuel_price = st.number_input(
                "Fuel Price ($/MMBtu)",
                value=float(query_params.get("fuel_price", DEFAULTS_OM['fuel_price_dollar_per_mmbtu'])),
                format="%.2f",
                key="fuel_price",
                on_change=update_param,
                args=("fuel_price",)
            )
            solar_om_fixed = st.number_input(
                "Solar Fixed O&M ($/kW)",
                value=int(query_params.get("solar_om", DEFAULTS_OM['solar_fixed_dollar_per_kw'])),
                format="%d",
                key="solar_om",
                on_change=update_param,
                args=("solar_om",)
            )
            bess_om_fixed = st.number_input(
                "BESS Fixed O&M ($/kW)",
                value=float(query_params.get("bess_om", DEFAULTS_OM['bess_fixed_dollar_per_kw'])),
                format="%.1f",
                key="bess_om",
                on_change=update_param,
                args=("bess_om",)
            )
            generator_om_fixed = st.number_input(
                "Generator Fixed O&M ($/kW)", 
                value=float(query_params.get("gen_om_fixed", gen_config['opex']['fixed_om'])),
                format="%.2f",
                key="gen_om_fixed",
                on_change=update_param,
                args=("gen_om_fixed",)
            )
            generator_om_variable = st.number_input(
                "Generator Variable O&M ($/kWh)", 
                value=float(query_params.get("gen_om_var", gen_config['opex']['variable_om'])),
                format="%.3f",
                key="gen_om_var",
                on_change=update_param,
                args=("gen_om_var",)
            )
            bos_om_fixed = st.number_input(
                "Balance of System Fixed O&M ($/kW-load)",
                value=float(query_params.get("bos_om", DEFAULTS_OM['bos_fixed_dollar_per_kw_load'])),
                format="%.1f",
                key="bos_om",
                on_change=update_param,
                args=("bos_om",)
            )
            soft_om_pct = st.number_input(
                "Soft O&M (% of hard capex)",
                value=float(query_params.get("soft_om", DEFAULTS_OM['soft_pct'])),
                format="%.2f",
                key="soft_om",
                on_change=update_param,
                args=("soft_om",)
            )
            
        # Column 2: System-wide O&M and Escalators
        with col2:
            st.subheader("Escalators")
            om_escalator = st.number_input(
                "O&M Escalator (% p.a.)",
                value=float(query_params.get("om_escalator", DEFAULTS_OM['escalator_pct'])),
                format="%.2f",
                key="om_escalator",
                on_change=update_param,
                args=("om_escalator",)
            )
            fuel_escalator = st.number_input(
                "Fuel Escalator (% p.a.)",
                value=float(query_params.get("fuel_escalator", DEFAULTS_OM['fuel_escalator_pct'])),
                format="%.2f",
                key="fuel_escalator",
                on_change=update_param,
                args=("fuel_escalator",)
            )

    return {
        'generator_om_fixed_dollar_per_kw': generator_om_fixed,
        'generator_om_variable_dollar_per_kwh': generator_om_variable,
        'fuel_price_dollar_per_mmbtu': fuel_price,
        'fuel_escalator_pct': fuel_escalator,
        'solar_om_fixed_dollar_per_kw': solar_om_fixed,
        'bess_om_fixed_dollar_per_kw': bess_om_fixed,
        'bos_om_fixed_dollar_per_kw_load': bos_om_fixed,
        'soft_om_pct': soft_om_pct,
        'om_escalator_pct': om_escalator,
        'cost_of_debt_pct': cost_of_debt,
        'leverage_pct': leverage,
        'debt_term_years': debt_term,
        'cost_of_equity_pct': cost_of_equity,
        'investment_tax_credit_pct': investment_tax_credit_pct,
        'combined_tax_rate_pct': combined_tax_rate,
        'construction_time_years': DEFAULTS_FINANCIAL['construction_time_years'],
        'depreciation_schedule': edited_depreciation['Depreciation (%)'].tolist(),
        # Solar PV CAPEX
        'capex_pv_modules': pv_modules,
        'capex_pv_inverters': pv_inverters,
        'capex_pv_racking': pv_racking,
        'capex_pv_balance_system': pv_balance_system,
        'capex_pv_labor': pv_labor,
        # BESS CAPEX
        'capex_bess_units': bess_units,
        'capex_bess_balance_of_system': bess_balance_of_system,
        'capex_bess_labor': bess_labor,
        # Generator CAPEX
        'capex_gensets': gensets,
        'capex_gen_balance_of_system': gen_balance_of_system,
        'capex_gen_labor': gen_labor,
        # System Integration CAPEX
        'capex_si_microgrid': si_microgrid,
        'capex_si_controls': si_controls,
        'capex_si_labor': si_labor,
        # Soft Costs
        'capex_soft_costs_general_conditions': soft_costs_general_conditions,
        'capex_soft_costs_epc_overhead': soft_costs_epc_overhead,
        'capex_soft_costs_design_engineering': soft_costs_design_engineering,
        'capex_soft_costs_permitting': soft_costs_permitting,
        'capex_soft_costs_startup': soft_costs_startup,
        'capex_soft_costs_insurance': soft_costs_insurance,
        'capex_soft_costs_taxes': soft_costs_taxes
    }
