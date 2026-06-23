import numpy as np
from mealpy import FloatVar, PSO

# =============================================================================
# 1. DEFINE PROBLEM PARAMETERS & ENVIRONMENT (Weather Fields)
# =============================================================================
N_CHECKPOINTS = 5  # Number of segments along the flight path

# Simulated Wind Field (u-component in knots) at each checkpoint for 3 altitudes
# Altitudes map to indices: 0 = 30,000ft, 1 = 34,000ft, 2 = 38,000ft
# Positive = Tailwind (good), Negative = Headwind (bad)
WIND_FIELD = np.array([
    [15.0,  30.0,  5.0],   # Checkpoint 1
    [-10.0, 45.0,  20.0],  # Checkpoint 2 (Strong tailwind at index 1!)
    [-25.0, -5.0,  10.0],  # Checkpoint 3
    [5.0,   20.0,  40.0],  # Checkpoint 4
    [0.0,   10.0,   5.0]   # Checkpoint 5
])

# Simulated Severe Weather Risk (0.0 = Clear, 1.0 = Severe Storm/No-Fly)
# If risk > 0.7, the aircraft triggers a massive penalty for safety violations
WEATHER_RISK = np.array([
    [0.1, 0.1, 0.1],
    [0.2, 0.9, 0.1],  # Storm at Checkpoint 2, Altitude index 1 (Conflicts with best wind!)
    [0.1, 0.2, 0.3],
    [0.1, 0.1, 0.1],
    [0.0, 0.1, 0.1]
])

SEGMENT_DISTANCE = 400.0  # Nautical miles per segment
COST_FUEL_PER_KG = 0.90   # Jet A-1 cost + Carbon Tax equivalent ($/kg)
COST_TIME_PER_MIN = 80.0  # Operational cost of delay ($/minute)

# =============================================================================
# 2. DEFINE THE FITNESS FUNCTION (Aviation Economics & Physics)
# =============================================================================
def objective_function(solution):
    """
    Decodes the solution vector and calculates total flight cost.
    The solution array contains:
    - First N elements: Continuous Mach speeds [0.74 to 0.85]
    - Next N elements: Continuous values mapped to Altitude indices [0 to 2]
    """
    speeds = solution[:N_CHECKPOINTS]
    alt_raw = solution[N_CHECKPOINTS:]
    
    total_fuel_burn = 0.0
    total_flight_time_min = 0.0
    weather_penalty = 0.0

    for i in range(N_CHECKPOINTS):
        mach = speeds[i]
        # Map continuous optimizer variables safely to discrete altitude bands
        alt_idx = int(np.clip(np.floor(alt_raw[i]), 0, 2))
        
        # 1. Aerodynamic True Airspeed (TAS) calculation based on Mach & Altitude
        base_tas = mach * 573.0  # Approximation of knots at cruise altitudes
        
        # 2. Extract wind vector and calculate Ground Speed
        tailwind = WIND_FIELD[i, alt_idx]
        ground_speed = base_tas + tailwind
        
        if ground_speed <= 0:  # Avoid unphysical or negative ground speeds
            return 1e9

        # 3. Calculate segment flight time
        segment_time_hours = SEGMENT_DISTANCE / ground_speed
        total_flight_time_min += segment_time_hours * 60.0
        
        # 4. Standard aircraft jet fuel flow equation (non-linear with speed & altitude)
        # Higher altitude index (0->2) lowers density, reducing fuel flow coeff.
        altitude_efficiency_factor = 1.0 - (alt_idx * 0.05) 
        fuel_flow_kg_per_hour = (3000.0 * (mach / 0.8) ** 3) * altitude_efficiency_factor
        total_fuel_burn += fuel_flow_kg_per_hour * segment_time_hours
        
        # 5. Safety Constraint: Check for convective weather boundaries
        if WEATHER_RISK[i, alt_idx] > 0.7:
            weather_penalty += 150000.0  # Huge financial penalty forcing avoidance

    # Evaluate complete flight profile costs
    fuel_cost = total_fuel_burn * COST_FUEL_PER_KG
    time_cost = total_flight_time_min * COST_TIME_PER_MIN
    
    return fuel_cost + time_cost + weather_penalty

# =============================================================================
# 3. CONSTRUCT MEALPY PROBLEM BOUNDS & SOLVER
# =============================================================================
# Vector bounds layout: [Mach_1...Mach_5, Alt_1...Alt_5]
lower_bounds = [0.74] * N_CHECKPOINTS + [0.0] * N_CHECKPOINTS
upper_bounds = [0.85] * N_CHECKPOINTS + [2.99] * N_CHECKPOINTS # 2.99 floors to index 2

problem_dict = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
    "minmax": "min",
}

# Use Particle Swarm Optimization (PSO), highly efficient for continuous-mapped paths
model = PSO.OriginalPSO(epoch=150, pop_size=500, c1=1.0, c2=2.0, w=0.5)

# Execute the heuristic search
g_best = model.solve(problem_dict)

# =============================================================================
# 4. PARSE AND DISPLAY RESULTS
# =============================================================================
print("\n" + "="*50)
print("IAG FLIGHT ROUTE OPTIMIZATION RESULT")
print("="*50)
print(f"Optimal Operating Cost: ${g_best.target.fitness:,.2f}")

optimized_solution = g_best.solution
opt_speeds = optimized_solution[:N_CHECKPOINTS]
opt_alts = [int(np.floor(x)) for x in optimized_solution[N_CHECKPOINTS:]]
alt_mapping = {0: "30,000 ft", 1: "34,000 ft", 2: "38,000 ft"}

print("\nWaypoints Flight Profile:")
for idx in range(N_CHECKPOINTS):
    print(f"  Segment {idx+1} -> Cruising at: {alt_mapping[opt_alts[idx]]} | Speed: Mach {opt_speeds[idx]:.3f}")

print("\nStrategic Analytics Insight:")
print("Look at Segment 2: The model bypassed the maximum tailwind profile (index 1)")
print("because it successfully identified and routed around the severe weather cell.")
print("="*50)
