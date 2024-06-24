#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm> // Para std::shuffle
#include <random>    // Para std::default_random_engine

#define NUM_CITIES 10
#define POPULATION_SIZE 1000  // Incrementar para mayor carga de trabajo

struct Individual {
    int cost;
    std::vector<int> path;

    Individual() : cost(0), path(NUM_CITIES) {}
    Individual(int cost, const std::vector<int>& path) : cost(cost), path(path) {}

    // Función para combinar dos instancias de Individual (reducción personalizada)
    static Individual combine(const Individual& a, const Individual& b) {
        if (a.cost < b.cost) {
            return a;
        } else {
            return b;
        }
    }
};

std::vector<std::vector<int>> cities(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<Individual> population(POPULATION_SIZE);

// Función para generar ciudades aleatorias
void generate_random_cities() {
    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            if (i != j) {
                cities[i][j] = rand() % 100 + 1;
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
    // Agregar el costo de regreso al inicio
    total_cost += cities[path[NUM_CITIES - 1]][path[0]];
    return total_cost;
}

// Función para inicializar la población de individuos
void initialize_population() {
    unsigned seed = 42;  // Semilla fija
    std::default_random_engine rng(seed);

    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].path.resize(NUM_CITIES);
        for (int j = 0; j < NUM_CITIES; ++j) {
            population[i].path[j] = j;
        }
        std::shuffle(population[i].path.begin(), population[i].path.end(), rng);
        population[i].cost = evaluate_cost(population[i].path);
    }
}

// Función para seleccionar una nueva generación de individuos (implementación ficticia)
std::vector<Individual> select_new_generation(const std::vector<Individual>& population, const std::vector<int>& costs) {
    // Implementación del algoritmo de selección
    // Por simplicidad, retornamos la misma población en este ejemplo
    return population;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42);  // Semilla fija

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
    for (int generation = 0; generation < 1000; ++generation) {
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            costs[i] = evaluate_cost(population[i].path);
        }
        population = select_new_generation(population, costs);
    }

    // Encontrar el mejor individuo local
    Individual local_best_individual = population[0];
    #pragma omp parallel for reduction(min : local_best_individual)
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (population[i].cost < local_best_individual.cost) {
            local_best_individual = population[i];
        }
    }

    // Recoger todos los mejores individuos locales en el proceso 0
    std::vector<Individual> all_best_individuals(size);
    MPI_Gather(&local_best_individual, sizeof(Individual), MPI_BYTE, all_best_individuals.data(), sizeof(Individual), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Encontrar el mejor individuo global en el proceso 0
    Individual global_best_individual;
    if (rank == 0) {
        global_best_individual = all_best_individuals[0];
        for (int i = 1; i < size; ++i) {
            if (all_best_individuals[i].cost < global_best_individual.cost) {
                global_best_individual = all_best_individuals[i];
            }
        }

        // Medición del tiempo de ejecución
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Imprimir resultados
        std::cout << "Mejor individuo (costo): " << global_best_individual.cost << "\n";
        std::cout << "Camino: ";
        for (int city : global_best_individual.path) {
            std::cout << city << " ";
        }
        std::cout << "\n";

        // Imprimir tiempo de ejecución
        std::cout << "Tiempo de ejecución: " << elapsed_time << " milisegundos\n";
    }

    MPI_Finalize();
    return 0;
}
