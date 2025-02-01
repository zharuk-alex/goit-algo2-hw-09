import random
import math
import numpy as np


def sphere_function(x):
    return sum(xi**2 for xi in x)


def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    def get_neighbors(current, step_size=0.1):
        x, y = current
        return [
            (x + step_size, y),
            (x - step_size, y),
            (x, y + step_size),
            (x, y - step_size),
        ]

    current_point = tuple(
        random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))
    )
    current_value = func(current_point)

    for _ in range(iterations):
        neighbors = get_neighbors(current_point, step_size=0.1)

        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            if all(
                bounds[i][0] <= neighbor[i] <= bounds[i][1] for i in range(len(bounds))
            ):
                value = func(neighbor)
                if value < next_value:
                    next_point = neighbor
                    next_value = value

        if next_value >= current_value or abs(next_value - current_value) < epsilon:
            break

        current_point, current_value = next_point, next_value

    return current_point, current_value


def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    def get_random_neighbor(current, step_size=0.5):
        x, y = current
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        return (new_x, new_y)

    current_point = (
        random.uniform(bounds[0][0], bounds[0][1]),
        random.uniform(bounds[1][0], bounds[1][1]),
    )
    current_value = func(current_point)

    for _ in range(iterations):
        new_point = get_random_neighbor(current_point, step_size=0.5)
        new_value = func(new_point)

        if abs(new_value - current_value) < epsilon:
            break

        if new_value < current_value or random.random() < 0.2:
            current_point, current_value = new_point, new_value

    return current_point, current_value


def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    def generate_neighbor(solution):
        x, y = solution
        new_x = x + random.uniform(-1, 1)
        new_y = y + random.uniform(-1, 1)
        return (new_x, new_y)

    current_solution = (
        random.uniform(bounds[0][0], bounds[0][1]),
        random.uniform(bounds[1][0], bounds[1][1]),
    )
    current_energy = func(current_solution)

    for _ in range(iterations):
        new_solution = generate_neighbor(current_solution)
        new_energy = func(new_solution)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
            current_solution = new_solution
            current_energy = new_energy

        temp *= cooling_rate

        if temp < epsilon:
            break

    return current_solution, current_energy


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
