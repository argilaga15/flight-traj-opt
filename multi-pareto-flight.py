import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# =============================================================================
# 1. ENVIRONMENT & SIMULATION PARAMETERS
# =============================================================================
N_CHECKPOINTS = 5
SEGMENT_DISTANCE = 400.0  # Nautical miles per segment

# Wind and Weather Data (Same as previous setup)
WIND_FIELD = np.array([
    [15.0,  30.0,  5.0],   
    [-10.0, 45.0,  20.0],  # Strong tailwind at index 1
    [-25.0, -5.0,  10.0],  
    [5.0,   20.0,  40.0],  
    [0.0,   10.0,   5.0]   
])

WEATHER_RISK = np.array([
    [0.1, 0.1, 0.1],
    [0.2, 0.9, 0.1],  # Severe storm at checkpoint 2, altitude index 1
    [0.1, 0.2, 0.3],
    [0.1, 0.1, 0.1],
    [0.0, 0.1, 0.1]
])

# =============================================================================
# 2. FITNESS FUNCTION (DEAP multi-objective requires returning a tuple)
# =============================================================================
def evaluate_flight(individual):
    """
    Evaluates a flight profile.
    Returns: (Total Fuel Burn in kg, Total Flight Time in minutes)
    Both objectives are to be MINIMIZED.
    """
    speeds = individual[:N_CHECKPOINTS]
    alt_raw = individual[N_CHECKPOINTS:]
    
    total_fuel_burn = 0.0
    total_flight_time_min = 0.0
    weather_penalty_fuel = 0.0
    weather_penalty_time = 0.0

    for i in range(N_CHECKPOINTS):
        mach = speeds[i]
        alt_idx = int(np.clip(np.floor(alt_raw[i]), 0, 2))
        
        # Physics / Aerodynamics
        base_tas = mach * 573.0  
        tailwind = WIND_FIELD[i, alt_idx]
        ground_speed = base_tas + tailwind
        
        if ground_speed <= 100:  # Avoid stalling or unrealistic speeds
            return 999999.0, 999999.0

        segment_time_hours = SEGMENT_DISTANCE / ground_speed
        total_flight_time_min += segment_time_hours * 60.0
        
        altitude_efficiency = 1.0 - (alt_idx * 0.05) 
        fuel_flow = (3000.0 * (mach / 0.8) ** 3) * altitude_efficiency
        total_fuel_burn += fuel_flow * segment_time_hours
        
        # Hard Weather Penalty: Adds massive values to both objectives if violated
        if WEATHER_RISK[i, alt_idx] > 0.7:
            weather_penalty_fuel += 50000.0
            weather_penalty_time += 500.0

    return total_fuel_burn + weather_penalty_fuel, total_flight_time_min + weather_penalty_time

# =============================================================================
# 3. DEAP SETUP FOR NSGA-II
# =============================================================================
# Define a minimizing fitness for 2 objectives: (Fuel, Time)
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual():
    # Genotype: [Mach_1...Mach_5, Alt_1...Alt_5]
    mach_speeds = [random.uniform(0.74, 0.85) for _ in range(N_CHECKPOINTS)]
    altitudes = [random.uniform(0.0, 2.99) for _ in range(N_CHECKPOINTS)]
    return creator.Individual(mach_speeds + altitudes)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_flight)

# Genetic operators optimized for continuous spaces
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[0.74]*5 + [0.0]*5, up=[0.85]*5 + [2.99]*5, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.74]*5 + [0.0]*5, up=[0.85]*5 + [2.99]*5, eta=20.0, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# =============================================================================
# 4. EXECUTE EVOLUTIONARY ALGORITHM
# =============================================================================
def main():
    random.seed(42)
    POP_SIZE = 100
    NGEN = 50  # Number of generations
    CXPB = 0.8  # Crossover probability
    MUTPB = 0.2 # Mutation probability
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Track the absolute best Pareto Front throughout history
    pareto_front = tools.ParetoFront()
    
    # Run the standard NSGA-II loop
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, 
                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                              stats=None, halloffame=pareto_front, verbose=False)
    
    # Extract objective values from valid Pareto routes
    fuel_coords = [ind.fitness.values[0] for ind in pareto_front if ind.fitness.values[0] < 100000]
    time_coords = [ind.fitness.values[1] for ind in pareto_front if ind.fitness.values[1] < 5000]
    
    # Print out a slice of strategic choices for the operations room
    print("="*60)
    print("IAG OPERATIONS RESEARCH: PARETO TRAJECTORY SAMPLES")
    print("="*60)
    sorted_front = sorted(pareto_front, key=lambda x: x.fitness.values[1]) # Sort by time
    
    # Fastest Profile
    fastest = sorted_front[0]
    print(f"🥇 FULL THROTTLE STRATEGY (Fastest):")
    print(f"   Time: {fastest.fitness.values[1]:.1f} mins | Fuel: {fastest.fitness.values[0]:,.1f} kg")
    
    # Greenest Profile
    greenest = sorted_front[-1]
    print(f"\n🌱 ECO STRATEGY (Most Fuel Efficient):")
    print(f"   Time: {greenest.fitness.values[1]:.1f} mins | Fuel: {greenest.fitness.values[0]:,.1f} kg")
    print("="*60)

    # =============================================================================
    # 5. PLOT THE PARETO FRONT
    # =============================================================================
    plt.figure(figsize=(9, 5))
    plt.scatter(time_coords, fuel_coords, color="crimson", edgecolors="black", s=40, zorder=3)
    plt.plot(sorted(time_coords), sorted(fuel_coords, reverse=True), color="gray", linestyle="--", alpha=0.7)
    
    plt.title("IAG Multi-Objective Flight Optimization: Pareto Front", fontsize=12, fontweight="bold")
    plt.xlabel("Flight Duration (Minutes)", fontsize=10)
    plt.ylabel("Total Fuel Burned (Kilograms)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Highlight the trade-off zones
    plt.annotate('Minimum Time / High Fuel', xy=(min(time_coords), max(fuel_coords)), 
                 xytext=(min(time_coords)+2, max(fuel_coords)-200),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate('Eco-Cruise / Higher Time', xy=(max(time_coords), min(fuel_coords)), 
                 xytext=(max(time_coords)-12, min(fuel_coords)+400),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
