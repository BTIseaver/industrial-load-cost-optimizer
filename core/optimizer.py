"""Fast coarse-to-fine optimizer for solar-plus-storage system sizing.

The candidate dispatch kernel follows the hourly battery and generator rules in
``core.powerflow_model``. It only stores the annual generation and served-load
totals needed to screen candidates; the winning configuration is rerun through
the full original simulator before results are displayed.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit

from core.defaults import BESS_HRS_STORAGE, GENERATOR_HEAT_RATES

BATTERY_ROUND_TRIP_EFFICIENCY = 0.92
BATTERY_DEGRADATION_PCT_PER_YEAR = 0.35 / 20
SOLAR_DEGRADATION_PCT_PER_YEAR = 0.005
DC_AC_RATIO = 1.2
SYSTEM_LIFETIME_YEARS = 20

_SQRT_BATTERY_EFFICIENCY = math.sqrt(BATTERY_ROUND_TRIP_EFFICIENCY)


@njit(cache=True, nogil=True)
def _annual_dispatch_metrics(
    solar_profile: np.ndarray,
    solar_capacity_mw: float,
    bess_power_mw: float,
    generator_capacity_mw: float,
    load_mw: float,
) -> np.ndarray:
    """Return rounded annual generator output and load served for 20 years."""
    annual = np.zeros((SYSTEM_LIFETIME_YEARS, 2), dtype=np.float64)
    battery_capacity_mwh = bess_power_mw * BESS_HRS_STORAGE

    for operating_year in range(1, SYSTEM_LIFETIME_YEARS + 1):
        degraded_capacity_mwh = battery_capacity_mwh * (
            1.0 - BATTERY_DEGRADATION_PCT_PER_YEAR * (operating_year - 1)
        )
        state = battery_capacity_mwh
        solar_factor = (
            (solar_capacity_mw / DC_AC_RATIO)
            * (1.0 - SOLAR_DEGRADATION_PCT_PER_YEAR * (operating_year - 1))
        )
        generator_total = 0.0
        unmet_total = 0.0

        for hour in range(len(solar_profile)):
            solar_mw = solar_profile[hour] * solar_factor
            power_balance = solar_mw - load_mw

            if power_balance > 0.0:
                excess_power = power_balance
                available_storage = degraded_capacity_mwh - state
                stored_energy = min(
                    min(excess_power, bess_power_mw), available_storage
                )
                state += stored_energy * _SQRT_BATTERY_EFFICIENCY
            else:
                deficit_power = -power_balance
                max_discharge = min(
                    bess_power_mw,
                    min(
                        deficit_power / _SQRT_BATTERY_EFFICIENCY,
                        state,
                    ),
                )
                battery_discharge = max_discharge * _SQRT_BATTERY_EFFICIENCY
                remaining_deficit = deficit_power - battery_discharge
                generator_total += min(remaining_deficit, generator_capacity_mw)
                unmet_total += max(remaining_deficit - generator_capacity_mw, 0.0)
                state -= max_discharge

        annual[operating_year - 1, 0] = round(generator_total)
        annual[operating_year - 1, 1] = round(load_mw * 8760.0 - unmet_total)

    return annual


def scan_capacity_pairs(
    solar_profile: Sequence[float],
    pairs: Sequence[Tuple[int, int]],
    load_mw: float,
    generator_capacity_mw: float,
) -> np.ndarray:
    """Dispatch a batch of Solar/BESS pairs.

    Output columns are Solar MW, BESS MW, 20 annual generator outputs, then 20
    annual served-load totals. Keeping just these 42 values makes the coarse
    grid cache compact while retaining all values used by the target and LCOE
    calculations.
    """
    profile = np.ascontiguousarray(solar_profile, dtype=np.float64)
    pair_array = np.asarray(pairs, dtype=np.int64)
    if pair_array.size == 0:
        return np.empty((0, 2 + SYSTEM_LIFETIME_YEARS * 2), dtype=np.float64)
    pair_array = pair_array.reshape((-1, 2))

    records = np.empty(
        (len(pair_array), 2 + SYSTEM_LIFETIME_YEARS * 2), dtype=np.float64
    )
    for index, (solar_mw, bess_mw) in enumerate(pair_array):
        annual = _annual_dispatch_metrics(
            profile,
            float(solar_mw),
            float(bess_mw),
            float(generator_capacity_mw),
            float(load_mw),
        )
        records[index, 0] = solar_mw
        records[index, 1] = bess_mw
        records[index, 2 : 2 + SYSTEM_LIFETIME_YEARS] = annual[:, 0]
        records[index, 2 + SYSTEM_LIFETIME_YEARS :] = annual[:, 1]
    return records


def capacity_search_limits(load_mw: float) -> Tuple[int, int, int]:
    """Choose load-scaled capacity limits and a coarse screening interval.

    The Solar and BESS input limits grow with load to permit optimization at
    seasonal-storage locations. They include at least the current 10 GW UI
    range and are rounded to 100 MW so every endpoint remains on the allowed
    fine grid.
    """
    load_mw = max(float(load_mw), 0.0)
    max_solar_mw = max(10_000, int(math.ceil(load_mw * 150 / 100) * 100))
    max_bess_mw = max(10_000, int(math.ceil(load_mw * 500 / 100) * 100))
    coarse_step_mw = max(
        1_000,
        int(math.ceil(max_bess_mw / 250 / 100) * 100),
    )
    return max_solar_mw, max_bess_mw, coarse_step_mw


def _grid_values(maximum: int, step: int) -> list[int]:
    values = list(range(0, maximum + 1, step))
    if values[-1] != maximum:
        values.append(maximum)
    return values


def _records_to_grid(
    records: np.ndarray,
    target_renewable_pct: float,
    load_mw: float,
    generator_capacity_mw: float,
    inputs: Dict,
    capex_subtotals: Dict,
    generator_type: str,
) -> list[dict]:
    feasible = []
    year_count = SYSTEM_LIFETIME_YEARS
    for record in records:
        gen_mwh = record[2 : 2 + year_count]
        served_mwh = record[2 + year_count :]
        served_total = float(served_mwh.sum())
        if served_total <= 0:
            continue
        renewable_pct = 100.0 * (1.0 - float(gen_mwh.sum()) / served_total)
        if renewable_pct + 1e-12 < target_renewable_pct:
            continue
        solar_mw = int(record[0])
        bess_mw = int(record[1])
        lcoe = estimate_lcoe_from_annual_totals(
            gen_mwh,
            served_mwh,
            solar_mw,
            bess_mw,
            generator_capacity_mw,
            load_mw,
            generator_type,
            inputs,
            capex_subtotals,
        )
        feasible.append(
            {
                "solar_mw": solar_mw,
                "bess_mw": bess_mw,
                "renewable_pct": renewable_pct,
                "lcoe": lcoe,
            }
        )
    return feasible


def estimate_lcoe_from_annual_totals(
    generator_mwh: Sequence[float],
    served_mwh: Sequence[float],
    solar_mw: int,
    bess_mw: int,
    generator_capacity_mw: float,
    load_mw: float,
    generator_type: str,
    inputs: Dict,
    capex_subtotals: Dict,
) -> float:
    """Solve the existing pro-forma LCOE equation algebraically.

    Electricity-price revenue enters the cash-flow model linearly, including
    its tax effect, so its zero-NPV price is ``-NPV(price=0) / dNPV_dprice``.
    This matches DataCenter.calculate_lcoe without building and iterating a
    pandas pro forma for every coarse candidate. The final candidate is always
    recalculated by the existing DataCenter implementation.
    """
    solar_capex = capex_subtotals["solar"]["rate"] * solar_mw
    bess_capex = (
        capex_subtotals["bess"]["rate"]
        * bess_mw
        * BESS_HRS_STORAGE
        / 1_000
    )
    generator_capex = (
        capex_subtotals["generator"]["rate"] * generator_capacity_mw / 1_000
    )
    integration_capex = (
        capex_subtotals["system_integration"]["rate"] * load_mw / 1_000
    )
    hard_capex = solar_capex + bess_capex + generator_capex + integration_capex
    soft_pct = capex_subtotals["soft_costs"]["rate"]
    total_capex = hard_capex * (1.0 + soft_pct / 100.0)

    leverage = float(inputs["leverage_pct"]) / 100.0
    total_debt = total_capex * leverage
    debt_rate = float(inputs["cost_of_debt_pct"]) / 100.0
    debt_term = int(inputs["debt_term_years"])
    if debt_rate == 0.0:
        fixed_debt_payment = total_debt / debt_term
    else:
        debt_growth = (1.0 + debt_rate) ** debt_term
        fixed_debt_payment = total_debt * debt_rate * debt_growth / (debt_growth - 1.0)

    hard_capex_nonzero = hard_capex != 0.0
    if hard_capex_nonzero:
        renewable_capex_fraction = (solar_capex + bess_capex) / hard_capex
    else:
        renewable_capex_fraction = 0.0
    tax_credit = (
        total_capex
        * renewable_capex_fraction
        * float(inputs["investment_tax_credit_pct"])
        / 100.0
    )
    depreciable_amount = total_capex - tax_credit / 2.0

    construction_years = int(inputs["construction_time_years"])
    cost_of_equity = float(inputs["cost_of_equity_pct"]) / 100.0
    tax_rate = float(inputs["combined_tax_rate_pct"]) / 100.0
    om_escalator = float(inputs["om_escalator_pct"]) / 100.0
    fuel_escalator = float(inputs["fuel_escalator_pct"]) / 100.0
    fuel_price = float(inputs["fuel_price_dollar_per_mmbtu"])
    solar_om = float(inputs["solar_om_fixed_dollar_per_kw"])
    bess_om = float(inputs["bess_om_fixed_dollar_per_kw"])
    generator_om = float(inputs["generator_om_fixed_dollar_per_kw"])
    generator_variable_om = float(inputs["generator_om_variable_dollar_per_kwh"])
    bos_om = float(inputs["bos_om_fixed_dollar_per_kw_load"])
    soft_om = float(inputs["soft_om_pct"])
    heat_rate = float(GENERATOR_HEAT_RATES[generator_type])
    depreciation_schedule = inputs["depreciation_schedule"]

    # Construction-period equity CAPEX is independent of the electricity price.
    npv_at_zero = 0.0
    capex_per_year = total_capex / construction_years
    construction_equity_capex = -capex_per_year * (1.0 - leverage)
    for year_index in range(-construction_years + 1, 1):
        discount_period = year_index + construction_years
        npv_at_zero += construction_equity_capex / (
            (1.0 + cost_of_equity) ** discount_period
        )

    price_derivative = 0.0
    debt_outstanding = total_debt
    for offset in range(SYSTEM_LIFETIME_YEARS):
        operating_year = offset + 1
        discount_period = operating_year + construction_years
        discount = (1.0 + cost_of_equity) ** discount_period
        gen_mwh = float(generator_mwh[offset])
        served = float(served_mwh[offset])
        om_factor = (1.0 + om_escalator) ** offset
        fuel_factor = (1.0 + fuel_escalator) ** offset

        generator_fuel_mmbtu = gen_mwh * heat_rate / 1_000.0
        fuel_cost = -fuel_price * fuel_factor * generator_fuel_mmbtu / 1_000_000.0
        solar_fixed_cost = -solar_om * om_factor * solar_mw * 1_000 / 1_000_000.0
        bess_fixed_cost = -bess_om * om_factor * bess_mw * 1_000 / 1_000_000.0
        generator_fixed_cost = (
            -generator_om * om_factor * generator_capacity_mw * 1_000 / 1_000_000.0
        )
        bos_fixed_cost = -bos_om * om_factor * load_mw * 1_000 / 1_000_000.0
        soft_om_cost = -soft_om * om_factor / 100.0 * hard_capex
        variable_om_cost = (
            -generator_variable_om * om_factor * gen_mwh * 1_000 / 1_000_000.0
        )
        operating_costs = (
            fuel_cost
            + solar_fixed_cost
            + bess_fixed_cost
            + generator_fixed_cost
            + bos_fixed_cost
            + soft_om_cost
            + variable_om_cost
        )

        if debt_outstanding is None:
            interest_expense = None
        else:
            interest_expense = -debt_outstanding * debt_rate
        debt_service = -fixed_debt_payment
        depreciation_rate = (
            float(depreciation_schedule[offset]) / 100.0
            if offset < len(depreciation_schedule)
            else 0.0
        )
        depreciation = -depreciation_rate * depreciable_amount

        # DataCenter produces NaN tax cash flow when its debt schedule has no
        # outstanding balance row; its after-tax cash-flow sum fills that NaN
        # with zero, so reproduce that behavior here.
        if interest_expense is None:
            tax_benefit = 0.0
            after_tax_cash_flow_at_zero = operating_costs + debt_service
            after_tax_price_factor = 1.0
        else:
            taxable_income_at_zero = operating_costs + depreciation + interest_expense
            federal_itc = tax_credit if operating_year == 1 else 0.0
            tax_benefit = -taxable_income_at_zero * tax_rate + federal_itc
            after_tax_cash_flow_at_zero = (
                operating_costs + debt_service + tax_benefit
            )
            after_tax_price_factor = 1.0 - tax_rate

        npv_at_zero += after_tax_cash_flow_at_zero / discount
        price_derivative += (
            served / 1_000_000.0 * after_tax_price_factor / discount
        )

        if operating_year < debt_term and debt_outstanding is not None:
            principal_payment = debt_service - interest_expense
            debt_outstanding += principal_payment
        else:
            debt_outstanding = None

    if price_derivative <= 0.0:
        return math.inf
    return -npv_at_zero / price_derivative


def _grid_candidates(records: np.ndarray, load_mw: float, target: float, **kwargs) -> list[dict]:
    results = _records_to_grid(
        records,
        target,
        load_mw,
        kwargs["generator_capacity_mw"],
        kwargs["inputs"],
        kwargs["capex_subtotals"],
        kwargs["generator_type"],
    )
    return results


def _select_seeds(candidates: list[dict], coarse_step_mw: int, count: int = 12) -> list[dict]:
    """Keep low-LCOE coarse candidates while spreading seeds across the grid."""
    seeds: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["lcoe"]):
        if all(
            max(
                abs(candidate["solar_mw"] - seed["solar_mw"]),
                abs(candidate["bess_mw"] - seed["bess_mw"]),
            ) >= coarse_step_mw
            for seed in seeds
        ):
            seeds.append(candidate)
            if len(seeds) >= count:
                break
    if not seeds and candidates:
        seeds.append(min(candidates, key=lambda item: item["lcoe"]))
    return seeds


def optimize_design(
    solar_profile: Sequence[float],
    load_mw: float,
    target_renewable_pct: float,
    generator_type: str,
    inputs: Dict,
    capex_subtotals: Dict,
    max_solar_mw: int,
    max_bess_mw: int,
    coarse_step_mw: int,
    scan_function: Callable = scan_capacity_pairs,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> Dict:
    """Find a low-LCOE feasible design, screening coarsely then refining at 100 MW.

    The entire load-scaled range is screened at ``coarse_step_mw``. Up to 12
    well-spaced promising candidates are then searched within one coarse step
    in both directions using the app's 100 MW increments. All reported
    capacities are therefore on the 100 MW grid.
    """
    if load_mw <= 0:
        raise ValueError("Industrial load must be greater than zero to optimize.")
    if target_renewable_pct not in (95, 99):
        raise ValueError("The optimizer supports 95% and 99% targets.")

    gas_capacity = float(load_mw)
    solar_values = _grid_values(max_solar_mw, coarse_step_mw)
    bess_values = _grid_values(max_bess_mw, coarse_step_mw)
    coarse_pairs = [(solar, bess) for solar in solar_values for bess in bess_values]
    if progress_callback:
        progress_callback("screen", 0.05, f"Screening {len(coarse_pairs):,} coarse capacity pairs…")
    coarse_records = scan_function(
        solar_profile, coarse_pairs, load_mw, gas_capacity
    )
    if progress_callback:
        progress_callback("screen", 0.7, "Ranking feasible coarse candidates…")
    coarse_candidates = _grid_candidates(
        coarse_records,
        load_mw,
        target_renewable_pct,
        generator_capacity_mw=gas_capacity,
        inputs=inputs,
        capex_subtotals=capex_subtotals,
        generator_type=generator_type,
    )
    if coarse_candidates:
        seeds = _select_seeds(coarse_candidates, coarse_step_mw)
    else:
        # Refine the highest-renewable coarse designs before declaring a target
        # infeasible; the battery model has initialization/degradation details
        # that make a monotonicity assumption unsafe near the threshold.
        ranked_by_renewable = []
        for record in coarse_records:
            gen_mwh = record[2 : 2 + SYSTEM_LIFETIME_YEARS]
            served_mwh = record[2 + SYSTEM_LIFETIME_YEARS :]
            served_total = float(served_mwh.sum())
            renewable_pct = (
                100.0 * (1.0 - float(gen_mwh.sum()) / served_total)
                if served_total > 0
                else 0.0
            )
            ranked_by_renewable.append(
                {
                    "solar_mw": int(record[0]),
                    "bess_mw": int(record[1]),
                    "renewable_pct": renewable_pct,
                    "lcoe": math.inf,
                }
            )
        seeds = _select_seeds(
            sorted(ranked_by_renewable, key=lambda item: item["renewable_pct"], reverse=True),
            coarse_step_mw,
        )
    fine_pair_set = set()
    for seed in seeds:
        solar_start = max(0, seed["solar_mw"] - coarse_step_mw)
        solar_stop = min(max_solar_mw, seed["solar_mw"] + coarse_step_mw)
        bess_start = max(0, seed["bess_mw"] - coarse_step_mw)
        bess_stop = min(max_bess_mw, seed["bess_mw"] + coarse_step_mw)
        solar_fine = range(
            (solar_start // 100) * 100,
            (solar_stop // 100) * 100 + 1,
            100,
        )
        bess_fine = range(
            (bess_start // 100) * 100,
            (bess_stop // 100) * 100 + 1,
            100,
        )
        for solar in solar_fine:
            if solar <= max_solar_mw:
                for bess in bess_fine:
                    if bess <= max_bess_mw:
                        fine_pair_set.add((solar, bess))

    fine_pairs = sorted(fine_pair_set)
    if progress_callback:
        progress_callback(
            "refine", 0.75,
            f"Refining {len(fine_pairs):,} nearby candidates at 100 MW steps…",
        )
    fine_records = scan_function(solar_profile, fine_pairs, load_mw, gas_capacity)
    fine_candidates = _grid_candidates(
        fine_records,
        load_mw,
        target_renewable_pct,
        generator_capacity_mw=gas_capacity,
        inputs=inputs,
        capex_subtotals=capex_subtotals,
        generator_type=generator_type,
    )
    if not coarse_candidates and not fine_candidates:
        maximum_record = max(
            coarse_records,
            key=lambda record: 100.0
            * (1.0 - record[2 : 2 + SYSTEM_LIFETIME_YEARS].sum()
               / max(record[2 + SYSTEM_LIFETIME_YEARS :].sum(), 1.0)),
        )
        maximum_renewable = 100.0 * (
            1.0
            - maximum_record[2 : 2 + SYSTEM_LIFETIME_YEARS].sum()
            / max(maximum_record[2 + SYSTEM_LIFETIME_YEARS :].sum(), 1.0)
        )
        raise ValueError(
            f"No design reached {target_renewable_pct}% within the current search "
            f"limits. The highest coarse result was {maximum_renewable:.2f}% at "
            f"{int(maximum_record[0]):,} MW solar and {int(maximum_record[1]):,} MW BESS."
        )
    final_candidates = coarse_candidates + fine_candidates
    winner = min(final_candidates, key=lambda item: item["lcoe"])
    if progress_callback:
        progress_callback("complete", 1.0, "Best design found; running the full model for the result…")
    winner.update(
        {
            "target_renewable_pct": target_renewable_pct,
            "coarse_step_mw": coarse_step_mw,
            "coarse_pairs_evaluated": len(coarse_pairs),
            "refined_pairs_evaluated": len(fine_pairs),
            "search_method": "coarse scan with local 100 MW refinement",
        }
    )
    return winner
