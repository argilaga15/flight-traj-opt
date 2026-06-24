import numpy as np
import optuna
from pyswarms.single import GlobalBestPSO

# =====================================================
# FLIGHT MODEL
# =====================================================

START = np.array([0.0, 0.0])
END = np.array([1000.0, 0.0])

NUM_WAYPOINTS = 5

WEATHER_ZONE_CENTER = np.array([500.0, 0.0])
WEATHER_ZONE_RADIUS = 150.0

FUEL_WEIGHT = 1.0
TIME_WEIGHT = 0.3


# =====================================================
# COST FUNCTION
# =====================================================

def path_cost(position):
    """
    position shape:
    [x1,y1,x2,y2,...]
    """

    waypoints = position.reshape(NUM_WAYPOINTS, 2)

    full_path = np.vstack([
        START,
        waypoints,
        END
    ])

    total_distance = 0

    for i in range(len(full_path) - 1):
        total_distance += np.linalg.norm(
            full_path[i + 1] - full_path[i]
        )

    fuel_cost = total_distance

    cruise_speed = 850.0
    time_cost = total_distance / cruise_speed

    weather_penalty = 0

    for point in full_path:
        d = np.linalg.norm(
            point - WEATHER_ZONE_CENTER
        )

        if d < WEATHER_ZONE_RADIUS:
            weather_penalty += (
                WEATHER_ZONE_RADIUS - d
            ) * 50

    return (
        FUEL_WEIGHT * fuel_cost +
        TIME_WEIGHT * time_cost +
        weather_penalty
    )


def swarm_objective(X):
    """
    X shape:
    (n_particles, dimensions)
    """

    return np.array([
        path_cost(x)
        for x in X
    ])


# =====================================================
# PSO RUNNER
# =====================================================

def run_pso(
        w,
        c1,
        c2,
        swarm_size):

    dimensions = NUM_WAYPOINTS * 2

    bounds = (
        np.array([0] * dimensions),
        np.array([1000] * dimensions)
    )

    optimizer = GlobalBestPSO(
        n_particles=swarm_size,
        dimensions=dimensions,
        options={
            'c1': c1,
            'c2': c2,
            'w': w
        },
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        swarm_objective,
        iters=100,
        verbose=False
    )

    return best_cost


# =====================================================
# OPTUNA OBJECTIVE
# =====================================================

def objective(trial):

    w = trial.suggest_float(
        "w",
        0.3,
        0.95
    )

    c1 = trial.suggest_float(
        "c1",
        0.5,
        3.0
    )

    c2 = trial.suggest_float(
        "c2",
        0.5,
        3.0
    )

    swarm_size = trial.suggest_int(
        "swarm_size",
        20,
        150
    )

    return run_pso(
        w=w,
        c1=c1,
        c2=c2,
        swarm_size=swarm_size
    )


# =====================================================
# MAIN
# =====================================================

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=50
)

print("\nBEST PARAMETERS")
print(study.best_params)

print("\nBEST COST")
print(study.best_value)