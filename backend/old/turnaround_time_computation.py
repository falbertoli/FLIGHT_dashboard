import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# 1) Define the modified function 
#    - calculates "required tax credit" per gallon to fully offset 
#      the revenue drop, in $/gallon.
# ------------------------------------------------------------------------
def hydrogen_uti_rev(
    fraction_flights_year,
    tot_delta_flights_atl,
    flights_atl_to_dome,
    h2_demand_annual_gal,
    extra_turn_time,
    total_rev,
    baseline_jetA_util
):
    """
    Parameters
    ----------
    fraction_flights_year   : Fraction of flights using H2 in the given year
    tot_delta_flights_atl   : Total Delta flights at ATL (for scaling utilization impact)
    flights_atl_to_dome     : Ratio of flights from ATL that are domestic
    h2_demand_annual_gal    : Annual hydrogen demand in gallons
    extra_turn_time         : Additional turnaround time in minutes
    total_rev               : Total revenue in Millions of USD
    baseline_jetA_util      : Baseline Jet-A utilization in flight hours

    Returns
    -------
    utilization_h2          : Adjusted hydrogen utilization hours
    revenue_drop_m          : The drop in revenue (Millions USD)
    required_tax_crd_per_gal: The required tax credit in $/gal to fully
                              compensate for the revenue drop
    baseline_revenue_m      : Baseline revenue (Millions USD)
    new_h2_revenue_m        : New revenue (Millions USD) after adopting H2
    pct_drop                : Percentage drop in revenue
    """
    # Calculate utilization for hydrogen flights, accounting for extra turnaround time
    utilization_h2 = baseline_jetA_util - 2 * (
        fraction_flights_year
        * tot_delta_flights_atl
        * flights_atl_to_dome
        * (extra_turn_time / 60.0)
    )

    # Baseline revenue (if all flights at fraction_flights_year had no extra turn time)
    baseline_revenue_m = fraction_flights_year * flights_atl_to_dome * total_rev

    # New H2 revenue after losing some utilization
    if baseline_jetA_util != 0.0:
        new_h2_revenue_m = baseline_revenue_m * (utilization_h2 / baseline_jetA_util)
    else:
        new_h2_revenue_m = 0.0

    # Revenue drop
    revenue_drop_m = baseline_revenue_m - new_h2_revenue_m
    pct_drop = 0.0 if baseline_revenue_m == 0.0 else 100.0 * (revenue_drop_m / baseline_revenue_m)

    # ----------------------------------------------------------------
    # Required tax credit per gallon to fully offset revenue_drop_m
    #      (revenue_drop_m is in Millions $, h2_demand_annual_gal is gallons)
    #      Convert revenue_drop to $ (rather than Millions $) then divide by gallons
    # ----------------------------------------------------------------
    if h2_demand_annual_gal > 0:
        required_tax_crd_per_gal = (revenue_drop_m * 1_000_000) / h2_demand_annual_gal
    else:
        required_tax_crd_per_gal = 0.0

    return (
        utilization_h2, 
        revenue_drop_m, 
        required_tax_crd_per_gal, 
        baseline_revenue_m, 
        new_h2_revenue_m, 
        pct_drop
    )

# ------------------------------------------------------------------------
# 2) Main script
# ------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------
    # 2.1) Load and filter your data
    # -------------------------------
    uti_data = pd.read_csv('/Users/abelsare/Documents/GitHub/FLIGHT/Economic/utilization_data.csv')
    operations_data = pd.read_csv('/Users/abelsare/Documents/GitHub/FLIGHT/Economic/operations_data.csv')
    income_data = pd.read_csv('/Users/abelsare/Documents/GitHub/FLIGHT/Economic/income_data.csv')

    total_delta_uti = uti_data[
        (uti_data['UNIQUE_CARRIER'] == 'DL') & (uti_data['REGION'] == 'D')
    ]
    atl_delta_oper = operations_data[
        (operations_data['UNIQUE_CARRIER_NAME'] == 'Delta Air Lines Inc.') 
        & (operations_data['ORIGIN'] == 'ATL')
    ]
    total_delta_oper = operations_data[
        (operations_data['UNIQUE_CARRIER_NAME'] == 'Delta Air Lines Inc.')
    ]
    total_revenue = income_data[
        (income_data['UNIQUE_CARRIER_NAME'] == 'Delta Air Lines Inc.') 
        & (income_data['REGION'] == 'D')
    ]

    # -----------------------------------------------
    # 2.2) Baseline Inputs & Constants (for 2023)
    # -----------------------------------------------
    fraction_flights_2023 = float(input("Enter the fraction of flights using hydrogen (e.g., 0.8 for 80%): "))
    if fraction_flights_2023 <= 0.05:  
        fraction_flights_2023 = 0.05

    tot_delta_flights_atl = total_delta_oper['DEPARTURES_PERFORMED'].sum()
    flights_atl_to_dome = (
        atl_delta_oper['DEPARTURES_PERFORMED'].sum() 
        / tot_delta_flights_atl
    )

    # -- Calculate 2023 annual H2 demand in gallons (already multiplied by 7.48052)
    h2_demand_month_2023 = 250000 * 7.48052 #input from demand model
    h2_demand_annual_2023 = h2_demand_month_2023 * 12

    extra_turn_time_2023 = 30  # minutes
    total_rev_2023 = income_data['OP_REVENUES'].sum() / 1_000_000  # in Millions $
    baseline_jetA_util_2023 = (
        fraction_flights_2023 
        * flights_atl_to_dome 
        * total_delta_uti['REV_ACRFT_HRS_AIRBORNE_610'].sum()
    )

    # -----------------------------------------------
    # 2.3) USER INPUTS
    # -----------------------------------------------
    final_h2_year = int(input("Enter the final year by which hydrogen flights reach target fraction (e.g., 2040): "))
    fraction_flights_end_year = fraction_flights_2023

    start_year = 2023
    end_year = final_h2_year
    growth_rate = 0.02  # input from demand model
    years = range(start_year, end_year + 1)

    # Linear slope for the fraction of flights
    slope = (fraction_flights_end_year - fraction_flights_2023) / (final_h2_year - start_year)

    # -----------------------------------------------
    # 2.4) Turn-time decrease rate scenarios
    # -----------------------------------------------
    turn_time_decrease_rates = [0, 1, 2, 3, 4, 5]
    scenario_results = {}

    for rate in turn_time_decrease_rates:
        all_results = []

        for year in years:
            years_elapsed = year - start_year

            # 1) H2 fraction grows linearly until final_h2_year
            if year <= final_h2_year:
                fraction_flights_year = fraction_flights_2023 + slope * years_elapsed
            else:
                fraction_flights_year = fraction_flights_end_year

            # 2) Turnaround time
            this_year_turn_time = max(0, extra_turn_time_2023 - (rate * years_elapsed))

            # 3) Growth factor for scaling
            factor = (1 + growth_rate) ** years_elapsed

            # Scale up H2 demand, revenue, baseline utilization each year
            h2_demand_annual_scaled_gal   = h2_demand_annual_2023   * factor
            total_rev_scaled_m            = total_rev_2023          * factor
            baseline_jetA_util_scaled     = baseline_jetA_util_2023 * factor

            # Call the modified function
            (
                utilization_h2, 
                revenue_drop_m, 
                required_tax_crd_per_gal, 
                baseline_revenue_m, 
                new_h2_revenue_m, 
                pct_drop
            ) = hydrogen_uti_rev(
                fraction_flights_year, 
                tot_delta_flights_atl,
                flights_atl_to_dome, 
                h2_demand_annual_scaled_gal, 
                this_year_turn_time, 
                total_rev_scaled_m, 
                baseline_jetA_util_scaled
            )

            all_results.append({
                'Year'                     : year,
                'Growth Factor'            : factor,
                'Turn Time (min)'          : this_year_turn_time,
                'Fraction Flights (H2)'    : fraction_flights_year,
                'H2 Demand (annual, gal)'  : h2_demand_annual_scaled_gal,
                'Hydrogen Utilization'     : utilization_h2,
                'Baseline Revenue (M)'     : baseline_revenue_m,
                'Hydrogen Revenue (M)'     : new_h2_revenue_m,
                'Revenue Drop (M)'         : revenue_drop_m,
                'Pct Drop'                 : pct_drop,
                'Req. Tax Credit ($/gal)'  : required_tax_crd_per_gal
            })

        df_scenario = pd.DataFrame(all_results)
        scenario_results[rate] = df_scenario

        # Print the maximum required tax credit for this scenario
        max_credit = df_scenario['Req. Tax Credit ($/gal)'].max()
        print(f"Scenario: {rate} min/year reduction -> Max Required Tax Credit ($/gal) over {start_year}-{end_year}: {max_credit:,.2f}")

    # --------------------------------------------------------------------
    # Display the final DataFrame for a chosen scenario 
    # --------------------------------------------------------------------
    chosen_scenario = 3
    print(f"\n===== PROJECTION RESULTS FOR {chosen_scenario} min/year SCENARIO =====")
    print(scenario_results[chosen_scenario].to_string(index=False))

    # --------------------------------------------------------------------
    # Plot Revenue Drop (%) vs Year for each scenario
    # --------------------------------------------------------------------
    plt.figure(figsize=(10,6))

    for rate, df in scenario_results.items():
        plt.plot(
            df['Year'], 
            df['Pct Drop'], 
            marker='o', 
            label=f"{rate} min/year"
        )

    plt.title(f"Revenue Drop vs. Year for Different Turn-Time Reduction Rates\n(Linear H2 Adoption to {fraction_flights_end_year:.0%} by {final_h2_year})")
    plt.xlabel("Year")
    plt.ylabel("% Revenue Drop")
    plt.grid(True)
    plt.legend(title="Reduction Rate")
    plt.tight_layout()
    plt.show()

    print("===== END OF PROJECTION =====")
