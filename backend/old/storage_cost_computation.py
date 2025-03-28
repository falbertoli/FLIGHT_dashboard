import numpy as np
#import H2_demand_tool 1.py 

def calculate_h2_storage_cost(
    total_h2_volume_gal,        # Total hydrogen volume to store [gallons]
    number_of_tanks,            # Number of tanks
    tank_diameter_ft,           # Diameter of each cylindrical tank [ft]
    tank_length_ft,             # Length of each cylindrical tank [ft]
    cost_per_sqft_construction, # Construction cost [$/ft^2]
    cost_per_cuft_insulation    # Insulation material + manufacturing cost [$/ft^3]
):
    """
    Returns a dictionary of cost estimates for hydrogen storage infrastructure,
    
   total_h2_volume_gal:        Total hydrogen volume to store [gal].
   number_of_tanks:            Number of storage tanks.
   tank_diameter_ft:           Diameter of each cylindrical tank [ft].
   tank_length_ft:             Length of each cylindrical tank [ft].
   cost_per_sqft_construction: Construction cost in $/ft^2 (footprint).
   cost_per_cuft_insulation:   Insulation cost in $/ft^3.
   returns A dictionary with:
             - 'insulation_volume_total'
             - 'insulation_cost'
             - 'footprint_total'
             - 'construction_cost'
             - 'total_infrastructure_cost'
    """

    # --- 1) Convert total hydrogen volume from gallons to cubic feet
    #         (approx. 1 gal = 0.1337 ft^3)
    total_h2_volume_cuft = total_h2_volume_gal * 0.1337

    # --- 2) Compute the insulation volume for ALL tanks:
    #         Insulation Volume = π × (width/2)^2 × [2 × (h/d) - 1/3] × [V_total / N_tanks]
    #
    #     Where:
    #       width = tank_diameter_ft
    #       h/d   = (tank_length_ft / tank_diameter_ft)
    #       V_total is total_h2_volume_cuft
    #       N_tanks = number_of_tanks
    #
    #     
    h_over_d = tank_length_ft / tank_diameter_ft
    insulation_volume_total = (
        np.pi
        * (tank_diameter_ft / 2.0) ** 2
        * (2.0 * h_over_d - 1.0 / 3.0)
        * (total_h2_volume_cuft / number_of_tanks)
    )

    # --- 3) Insulation cost = total insulation volume × cost per cubic foot
    insulation_cost = insulation_volume_total * cost_per_cuft_insulation

    # --- 4) Footprint calculation:
    #         Assume each tank occupies (diameter × length) on the ground,
    #         and multiply by the number of tanks.
    footprint_per_tank_sqft = tank_diameter_ft * tank_length_ft
    footprint_total = footprint_per_tank_sqft * number_of_tanks

    # --- 5) Construction cost = footprint area × construction cost rate
    construction_cost = footprint_total * cost_per_sqft_construction

    # --- 6) Total infrastructure cost = insulation + construction
    total_infrastructure_cost = insulation_cost + construction_cost

    return {
        "insulation_volume_total": insulation_volume_total,
        "insulation_cost": insulation_cost,
        "footprint_total": footprint_total,
        "construction_cost": construction_cost,
        "total_infrastructure_cost": total_infrastructure_cost
    }


# ---------------------------

if __name__ == "__main__":
    # Example input assumptions (To be changed with values from demand tool)
    total_h2_volume_gal = 5000000   # gallons
    number_of_tanks = 20
    tank_diameter_ft = 10
    tank_length_ft = 40
    cost_per_sqft_construction = 580   # $/ft^2
    cost_per_cuft_insulation = 15     # $/ft^3
    
    results = calculate_h2_storage_cost(
        total_h2_volume_gal,
        number_of_tanks,
        tank_diameter_ft,
        tank_length_ft,
        cost_per_sqft_construction,
        cost_per_cuft_insulation
    )
    
    print("=== Hydrogen Storage Cost Estimation ===")
    for key, val in results.items():
        print(f"{key}: {val:,.2f}")
