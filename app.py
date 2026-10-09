"""Main Streamlit entrypoint for the industrial-load cost optimizer."""

from typing import Dict, Tuple

import numpy as np
import streamlit as st

from app_components.st_inputs import (
    calculate_capex_subtotals,
    create_financial_inputs,
    create_map_input,
    create_preset_controls,
    create_system_inputs,
)
from app_components.st_outputs import (
    create_capex_chart,
    create_energy_mix_chart,
    create_subcategory_capex_charts,
    display_daily_sample_chart,
    display_intro_section,
    display_proforma,
    format_proforma,
)
from core.datacenter import DataCenter
from core.optimizer import capacity_search_limits, optimize_design, scan_capacity_pairs
from core.powerflow_model import (
    SYSTEM_LIFETIME_YEARS,
    calculate_energy_mix,
    get_solar_ac_dataframe,
    simulate_system,
)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_capacity_scan(
    solar_profile: np.ndarray,
    pairs: Tuple[Tuple[int, int], ...],
    load_mw: float,
    generator_capacity_mw: float,
) -> np.ndarray:
    """Cache physical dispatch screens shared by targets and report modes."""
    return scan_capacity_pairs(solar_profile, pairs, load_mw, generator_capacity_mw)


def display_capex_breakdown(capex_subtotals: Dict) -> None:
    st.subheader("CAPEX Breakdown")
    total_capex = sum(component["total_absolute"] for component in capex_subtotals.values())
    st.metric("Total CAPEX", f"${total_capex:.1f}M")
    st.plotly_chart(create_capex_chart(capex_subtotals), use_container_width=True)
    with st.expander("Subcategory breakdown"):
        create_subcategory_capex_charts(capex_subtotals)


def display_energy_mix(energy_mix: Dict) -> None:
    st.subheader("Energy Mix")
    st.metric("Renewable share of served energy", f"{energy_mix['renewable_percentage']:.1f}%")
    st.plotly_chart(create_energy_mix_chart(energy_mix), use_container_width=True)


def _run_case(
    inputs: Dict,
    solar_dataframe,
    solar_capacity_mw: int,
    bess_power_mw: int,
    generator_capacity_mw: float,
    generator_type: str,
    remove_generator_costs: bool = False,
) -> Dict:
    """Run the full original powerflow and finance models for one design."""
    case_inputs = dict(inputs)
    case_inputs.update(
        {
            "solar_pv_capacity_mw": solar_capacity_mw,
            "bess_max_power_mw": bess_power_mw,
            "generator_capacity_mw": generator_capacity_mw,
            "generator_type": generator_type,
        }
    )
    if remove_generator_costs:
        case_inputs.update(
            {
                "capex_gensets": 0,
                "capex_gen_balance_of_system": 0,
                "capex_gen_labor": 0,
                "generator_om_fixed_dollar_per_kw": 0,
                "generator_om_variable_dollar_per_kwh": 0,
            }
        )

    powerflow = simulate_system(
        case_inputs["lat"],
        case_inputs["long"],
        solar_dataframe,
        solar_capacity_mw,
        bess_power_mw,
        generator_capacity_mw,
        case_inputs["datacenter_load_mw"],
    )
    annual_results = powerflow["annual_results"]
    capex_subtotals = calculate_capex_subtotals(case_inputs)

    data_center = DataCenter(
        solar_pv_capacity_mw=solar_capacity_mw,
        bess_max_power_mw=bess_power_mw,
        generator_capacity_mw=generator_capacity_mw,
        generator_type=generator_type,
        datacenter_load_mw=case_inputs["datacenter_load_mw"],
        solar_capex_total_dollar_per_w=capex_subtotals["solar"]["rate"],
        bess_capex_total_dollar_per_kwh=capex_subtotals["bess"]["rate"],
        generator_capex_total_dollar_per_kw=capex_subtotals["generator"]["rate"],
        system_integration_capex_total_dollar_per_kw=capex_subtotals["system_integration"]["rate"],
        soft_costs_capex_total_pct=capex_subtotals["soft_costs"]["rate"],
        om_solar_fixed_dollar_per_kw=case_inputs["solar_om_fixed_dollar_per_kw"],
        om_bess_fixed_dollar_per_kw=case_inputs["bess_om_fixed_dollar_per_kw"],
        om_generator_fixed_dollar_per_kw=case_inputs["generator_om_fixed_dollar_per_kw"],
        om_generator_variable_dollar_per_kwh=case_inputs["generator_om_variable_dollar_per_kwh"],
        fuel_price_dollar_per_mmbtu=case_inputs["fuel_price_dollar_per_mmbtu"],
        fuel_escalator_pct=case_inputs["fuel_escalator_pct"],
        om_bos_fixed_dollar_per_kw_load=case_inputs["bos_om_fixed_dollar_per_kw_load"],
        om_soft_pct=case_inputs["soft_om_pct"],
        om_escalator_pct=case_inputs["om_escalator_pct"],
        debt_term_years=case_inputs["debt_term_years"],
        leverage_pct=case_inputs["leverage_pct"],
        cost_of_debt_pct=case_inputs["cost_of_debt_pct"],
        cost_of_equity_pct=case_inputs["cost_of_equity_pct"],
        combined_tax_rate_pct=case_inputs["combined_tax_rate_pct"],
        investment_tax_credit_pct=case_inputs["investment_tax_credit_pct"],
        depreciation_schedule=case_inputs["depreciation_schedule"],
        construction_time_years=case_inputs["construction_time_years"],
        filtered_simulation_data=annual_results,
        location=f"{case_inputs['lat']},{case_inputs['long']}",
    )
    lcoe, pro_forma = data_center.calculate_lcoe()
    demand_mwh = (
        SYSTEM_LIFETIME_YEARS
        * 8760
        * case_inputs["datacenter_load_mw"]
    )
    load_served_pct = (
        100 * annual_results["Load Served (MWh)"].sum() / demand_mwh
        if demand_mwh > 0
        else 0.0
    )

    return {
        "daily_powerflow_results": powerflow["daily_sample"],
        "annual_powerflow_results": annual_results,
        "energy_mix": calculate_energy_mix(annual_results),
        "capex_subtotals": capex_subtotals,
        "lcoe": lcoe,
        "formatted_proforma": format_proforma(pro_forma),
        "load_served_pct": load_served_pct,
        "solar_capacity_mw": solar_capacity_mw,
        "bess_power_mw": bess_power_mw,
        "generator_capacity_mw": generator_capacity_mw,
        "generator_type": generator_type,
    }


def _render_completed_results(results: Dict) -> None:
    optimization = results.get("optimization")
    if optimization:
        st.subheader("Optimization Results")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Solar PV", f"{results['solar_capacity_mw']:,} MW")
        metric_cols[1].metric("BESS Power (4-hour)", f"{results['bess_power_mw']:,} MW")
        metric_cols[2].metric("Annual load served", f"{results['load_served_pct']:.2f}%")
        metric_cols[3].metric("Calculated LCOE", f"${results['lcoe']:.2f}/MWh")
        if optimization["standalone"]:
            st.caption(
                f"Sized to {optimization['target_renewable_pct']}% renewable output with a "
                f"{optimization['load_mw']:,.0f} MW gas turbine, then recalculated with the "
                "generator and its costs removed. Unserved demand has no outage penalty."
            )
        else:
            st.caption(
                f"Sized for at least {optimization['target_renewable_pct']}% renewable output "
                f"with a gas turbine equal to the {optimization['load_mw']:,.0f} MW load. "
                f"Coarse scan: {optimization['coarse_pairs_evaluated']:,} pairs; "
                f"100 MW refinement: {optimization['refined_pairs_evaluated']:,} pairs."
            )
    else:
        st.subheader("Levelized Cost of Electricity")
        st.metric("Calculated LCOE", f"${results['lcoe']:.2f}/MWh")

    st.divider()
    graph_col, energy_mix_col = st.columns([2, 2], gap="medium")
    with graph_col:
        st.subheader("Power Flow (Worst Week)")
        display_daily_sample_chart(results["daily_powerflow_results"])
    with energy_mix_col:
        display_energy_mix(results["energy_mix"])
        if not optimization:
            st.metric("Annual load served", f"{results['load_served_pct']:.2f}%")

    st.divider()
    display_capex_breakdown(results["capex_subtotals"])
    st.subheader("Financial Model")
    display_proforma(results["formatted_proforma"])


def main() -> None:
    display_intro_section()
    create_preset_controls()
    inputs = create_system_inputs()

    location_col, financial_col = st.columns([1, 1], gap="medium")
    with location_col:
        lat, longitude, location_name = create_map_input()
        inputs.update({"lat": lat, "long": longitude, "location_name": location_name})
    with financial_col:
        inputs.update(create_financial_inputs(inputs["generator_type"]))

    st.divider()
    run_calculation = st.button(
        "Run calculation",
        type="primary",
        help="Run the current manual configuration. Input changes alone do not start calculations.",
    )
    st.caption("Change inputs or apply presets, then press a button to run the model.")

    st.subheader("Optimize Solar PV and BESS")
    st.caption(
        "Each optimization sizes with a gas turbine equal to industrial load. "
        "Standalone options retain that optimized Solar/BESS design, then remove the turbine and its costs."
    )
    optimizer_columns = st.columns(4)
    optimize_95_gas = optimizer_columns[0].button("Optimize 95% with gas", use_container_width=True)
    optimize_99_gas = optimizer_columns[1].button("Optimize 99% with gas", use_container_width=True)
    optimize_95_standalone = optimizer_columns[2].button(
        "Optimize 95% without gas", use_container_width=True
    )
    optimize_99_standalone = optimizer_columns[3].button(
        "Optimize 99% without gas", use_container_width=True
    )

    optimize_action = None
    for pressed, target, standalone in (
        (optimize_95_gas, 95, False),
        (optimize_99_gas, 99, False),
        (optimize_95_standalone, 95, True),
        (optimize_99_standalone, 99, True),
    ):
        if pressed:
            optimize_action = (target, standalone)

    if run_calculation or optimize_action:
        try:
            with st.spinner(f"Fetching solar resource for {location_name}…"):
                solar_dataframe = get_solar_ac_dataframe(lat, longitude)

            if optimize_action:
                target, standalone = optimize_action
                load_mw = float(inputs["datacenter_load_mw"])
                if load_mw <= 0:
                    st.error("Set industrial load above zero before optimizing.")
                    return

                generator_type = "Gas Turbine"
                limits = capacity_search_limits(load_mw)
                progress_bar = st.progress(0.0, text="Preparing the capacity search…")

                def update_progress(_stage: str, amount: float, message: str) -> None:
                    progress_bar.progress(amount, text=message)

                search_inputs = dict(inputs)
                search_inputs["generator_type"] = generator_type
                gas_sized_inputs = dict(search_inputs)
                gas_sized_inputs["generator_capacity_mw"] = load_mw
                search_capex = calculate_capex_subtotals(gas_sized_inputs)
                profile = solar_dataframe["p_mp"].to_numpy(dtype=np.float64)

                with st.spinner(
                    f"Searching {limits[0]:,} MW solar by {limits[1]:,} MW BESS bounds…"
                ):
                    optimum = optimize_design(
                        profile,
                        load_mw,
                        target,
                        generator_type,
                        search_inputs,
                        search_capex,
                        limits[0],
                        limits[1],
                        limits[2],
                        scan_function=_cached_capacity_scan,
                        progress_callback=update_progress,
                    )

                generator_capacity = 0 if standalone else load_mw
                results = _run_case(
                    search_inputs,
                    solar_dataframe,
                    optimum["solar_mw"],
                    optimum["bess_mw"],
                    generator_capacity,
                    generator_type,
                    remove_generator_costs=standalone,
                )
                results["optimization"] = {
                    **optimum,
                    "standalone": standalone,
                    "load_mw": load_mw,
                }
                progress_bar.empty()
                st.session_state.calculation_results = results
                st.session_state.calculation_location = location_name
            else:
                results = _run_case(
                    inputs,
                    solar_dataframe,
                    inputs["solar_pv_capacity_mw"],
                    inputs["bess_max_power_mw"],
                    inputs["generator_capacity_mw"],
                    inputs["generator_type"],
                )
                st.session_state.calculation_results = results
                st.session_state.calculation_location = location_name
        except ValueError as error:
            st.error(str(error))
            return
        except Exception as error:
            st.error(f"Calculation failed: {error}")
            return

    if "calculation_results" not in st.session_state:
        st.info("Set the inputs above, then press Run calculation or an optimization button.")
        return
    if not run_calculation and not optimize_action:
        st.warning(
            f"Showing the last completed calculation for {st.session_state.get('calculation_location', 'the selected location')}. "
            "Press a run or optimization button to refresh the results."
        )
    _render_completed_results(st.session_state.calculation_results)


if __name__ == "__main__":
    main()
