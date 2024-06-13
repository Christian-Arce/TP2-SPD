#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>

#define NUM_CITIES 100
#define POPULATION_SIZE 10000
#define NUM_GENERATIONS 1000

struct Individual {
    int cost;
    std::vector<int> path;

    Individual() : cost(0), path(NUM_CITIES) {}
    Individual(int cost, const std::vector<int>& path) : cost(cost), path(path) {}
};

std::vector<std::vector<int>> cities(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<Individual> population(POPULATION_SIZE);

// Función para generar ciudades aleatorias
void generate_random_cities() {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            if (i != j) {
                cities[i][j] = dist(rng);
            } else {
                cities[i][j] = 0;
            }
        }
    }
}

// Función para evaluar el costo de un camino (ruta)
int evaluate_cost(const std::vector<int>& path) {
    int total_cost = 0;
    for (int i = 0; i < NUM_CITIES - 1; ++i) {
        int city_from = path[i];
        int city_to = path[i + 1];
        total_cost += cities[city_from][city_to];
    }
    total_cost += cities[path[NUM_CITIES - 1]][path[0]]; // Costo de regreso al inicio
    return total_cost;
}

// Función para inicializar la población de individuos
void initialize_population() {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<int> dist(0, NUM_CITIES - 1);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].path.resize(NUM_CITIES);
        for (int j = 0; j < NUM_CITIES; ++j) {
            population[i].path[j] = j;
        }
        std::shuffle(population[i].path.begin() + 1, population[i].path.end(), rng); // Shuffle sin cambiar el primer elemento (inicio)
        population[i].cost = evaluate_cost(population[i].path);
    }
}

// Función para seleccionar una nueva generación de individuos (implementación ficticia)
std::vector<Individual> select_new_generation(const std::vector<Individual>& population, const std::vector<int>& costs) {
    // Implementación de selección ficticia: mantener a los mejores individuos
    std::vector<Individual> new_generation(population.begin(), population.begin() + POPULATION_SIZE / 2);
    return new_generation;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank); // Semilla aleatoria basada en el rango para variar entre procesos

    auto start_time = std::chrono::steady_clock::now();

    if (rank == 0) {
        generate_random_cities();
    }

    // Broadcast de las ciudades generadas
    MPI_Bcast(&cities[0][0], NUM_CITIES * NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

    // Inicialización de la población
    initialize_population();

    // Ejemplo de cálculo de costos y selección
    std::vector<int> costs(POPULATION_SIZE);
    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        costs[i] = evaluate_cost(population[i].path);
    }

    // Recolección de resultados parciales de costos usando MPI_Gather
    std::vector<int> all_costs(POPULATION_SIZE * size);
    MPI_Gather(costs.data(), POPULATION_SIZE, MPI_INT, all_costs.data(), POPULATION_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Ejemplo de cálculo de nueva generación y selección
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        population = select_new_generation(population, costs);

        // Distribución de la nueva población entre procesos
        std::vector<Individual> local_population(population.begin() + rank * (POPULATION_SIZE / size), population.begin() + (rank + 1) * (POPULATION_SIZE / size));

        // Ejemplo de cálculo de costos y selección local
        std::vector<int> local_costs(local_population.size());
        #pragma omp parallel for
        for (int i = 0; i < local_population.size(); ++i) {
            local_costs[i] = evaluate_cost(local_population[i].path);
        }

        // Recolección de resultados parciales locales usando MPI_Gather
        std::vector<int> all_local_costs(local_population.size() * size);
        MPI_Gather(local_costs.data(), local_population.size(), MPI_INT, all_local_costs.data(), local_population.size(), MPI_INT, 0, MPI_COMM_WORLD);

        // Selección del mejor individuo local en cada proceso
        int local_best_index = std::min_element(local_costs.begin(), local_costs.end()) - local_costs.begin();
        Individual local_best_individual = local_population[local_best_index];

        // Recolección de los mejores individuos locales en proceso 0
        std::vector<int> all_best_costs(size);
        MPI_Gather(&local_best_individual.cost, 1, MPI_INT, all_best_costs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> local_best_path = local_best_individual.path;
        std::vector<int> all_best_paths(size * NUM_CITIES);

        MPI_Gather(local_best_path.data(), NUM_CITIES, MPI_INT, all_best_paths.data(), NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

        // Proceso 0 determina el mejor individuo global
        if (rank == 0) {
            int global_best_cost = all_best_costs[0];
            Individual global_best_individual = local_population[0];

            for (int i = 0; i < size; ++i) {
                if (all_best_costs[i] < global_best_cost) {
                    global_best_cost = all_best_costs[i];
                    global_best_individual = local_population[i];
                }
            }

            // Imprimir el mejor individuo global
            std::cout << "Mejor costo encontrado: " << global_best_individual.cost << std::endl;
            std::cout << "Mejor camino encontrado: ";
            for (int city : global_best_individual.path) {
                std::cout << city << " ";
            }
            std::cout << std::endl;
        }
    }

    // Medición del tiempo de ejecución
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (rank == 0) {
        // Imprimir tiempo de ejecución
        std::cout << "Tiempo de ejecución: " << elapsed_time << " milisegundos" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
