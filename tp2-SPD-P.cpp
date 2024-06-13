#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define NUM_CITIES 10
#define POPULATION_SIZE 100

struct Individual {
    int cost;
    std::vector<int> path;
};

std::vector<std::vector<int>> cities(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<Individual> population(POPULATION_SIZE);

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

void initialize_population() {
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].path.resize(NUM_CITIES);
        for (int j = 0; j < NUM_CITIES; ++j) {
            population[i].path[j] = j;
        }
        std::random_shuffle(population[i].path.begin(), population[i].path.end());
        population[i].cost = rand() % 1000;  // Ejemplo de costo aleatorio
    }
}

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

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    srand(time(0) + rank);  // Semilla aleatoria basada en el rango para variar entre procesos

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
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        costs[i] = population[i].cost;
    }

    for (int generation = 0; generation < 1000; ++generation) {
        population = select_new_generation(population, costs);
    }

    // Encontrar el mejor individuo local
    Individual local_best_individual = population[0];
    for (const auto& ind : population) {
        if (ind.cost < local_best_individual.cost) {
            local_best_individual = ind;
        }
    }

    // Recoger todos los mejores individuos locales en el proceso 0
    std::vector<Individual> all_best_individuals(size);
    MPI_Gather(&local_best_individual, sizeof(Individual), MPI_BYTE, &all_best_individuals[0], sizeof(Individual), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Encontrar el mejor individuo global en el proceso 0
    Individual global_best_individual;
    if (rank == 0) {
        global_best_individual = all_best_individuals[0];
        for (const auto& ind : all_best_individuals) {
            if (ind.cost < global_best_individual.cost) {
                global_best_individual = ind;
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
