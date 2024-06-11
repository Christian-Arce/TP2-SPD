#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <ctime>
#include <chrono>

#define NUM_CITIES 100  // Número de ciudades
#define POP_SIZE 10  // Tamaño de la población
#define NUM_GENERATIONS 100  // Número de generaciones
#define MUTATION_RATE 0.1  // Tasa de mutación

int cities[NUM_CITIES][NUM_CITIES];

typedef struct {
    std::vector<int> path;
    int cost;
} Individual;

std::vector<Individual> population(POP_SIZE);
std::vector<Individual> new_population(POP_SIZE);

// Función para calcular el costo de un camino
int calculate_cost(const std::vector<int>& path) {
    int cost = 0;
    for (int i = 0; i < NUM_CITIES - 1; i++) {
        cost += cities[path[i]][path[i + 1]];
    }
    cost += cities[path[NUM_CITIES - 1]][path[0]];  // Regresar al punto de partida
    return cost;
}

// Función para inicializar la población
void initialize_population() {
    std::random_device rd;
    std::mt19937 g(rd());
    #pragma omp parallel for
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].path.resize(NUM_CITIES);
        std::iota(population[i].path.begin(), population[i].path.end(), 0);  // Inicializa con 0, 1, ..., NUM_CITIES-1
        std::shuffle(population[i].path.begin(), population[i].path.end(), g);
        population[i].cost = calculate_cost(population[i].path);
    }
}

// Función para seleccionar dos individuos para reproducción (torneo)
int select_parents(const std::vector<int>& costs) {
    int tamTorneo = 3;  // Tamaño del torneo
    int mejor = rand() % POP_SIZE;
    for (int i = 1; i < tamTorneo; ++i) {
        int contendor = rand() % POP_SIZE;
        if (costs[contendor] < costs[mejor]) {
            mejor = contendor;
        }
    }
    return mejor;
}

// Función para hacer crossover entre dos padres
std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int start = rand() % NUM_CITIES;
    int end = start + rand() % (NUM_CITIES - start);
    
    std::vector<int> child(NUM_CITIES);
    std::vector<bool> visited(NUM_CITIES, false);
    
    for (int i = start; i <= end; i++) {
        child[i] = parent1[i];
        visited[child[i]] = true;
    }
    
    int pos = 0;
    for (int i = 0; i < NUM_CITIES; i++) {
        if (pos == start) pos = end + 1;
        if (!visited[parent2[i]]) {
            child[pos++] = parent2[i];
        }
    }
    return child;
}

// Función para mutar un camino
void mutate(std::vector<int>& path) {
    for (int i = 0; i < NUM_CITIES; i++) {
        if ((double)rand() / RAND_MAX < MUTATION_RATE) {
            int r = rand() % NUM_CITIES;
            std::swap(path[i], path[r]);
        }
    }
}

// Función para seleccionar la próxima generación
std::vector<Individual> select_new_generation(const std::vector<Individual>& population, const std::vector<int>& costs) {
    std::vector<Individual> new_generation(POP_SIZE);
    #pragma omp parallel for
    for (int i = 0; i < POP_SIZE / 2; ++i) {
        int parent1 = select_parents(costs);
        int parent2 = select_parents(costs);

        // Cruzar
        std::vector<int> child1 = crossover(population[parent1].path, population[parent2].path);
        std::vector<int> child2 = crossover(population[parent2].path, population[parent1].path);

        // Mutar
        mutate(child1);
        mutate(child2);

        // Calcular costo
        new_generation[2*i].path = child1;
        new_generation[2*i].cost = calculate_cost(child1);
        new_generation[2*i + 1].path = child2;
        new_generation[2*i + 1].cost = calculate_cost(child2);
    }
    return new_generation;
}

// Función para generar distancias aleatorias entre ciudades
void generate_random_cities() {
    for (int i = 0; i < NUM_CITIES; i++) {
        for (int j = 0; i < NUM_CITIES; i++) {
            if (i == j) {
                cities[i][j] = 0;
            } else {
                cities[i][j] = rand() % 100 + 1;  // Distancias aleatorias entre 1 y 100
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank);  // Semilla aleatoria basada en el rango para variar entre procesos

    if (rank == 0) {
        generate_random_cities();
    }

    // Broadcast de las ciudades generadas
    MPI_Bcast(&cities, NUM_CITIES * NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

    // Inicialización de la población
    initialize_population();

    // Almacenar los costos de los individuos
    std::vector<int> costs(POP_SIZE);

    // Medición del tiempo de ejecución
    auto start_time = std::chrono::steady_clock::now();

    // Bucle principal del algoritmo genético
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluación de costos
        #pragma omp parallel for
        for (int i = 0; i < POP_SIZE; ++i) {
            costs[i] = population[i].cost;
        }

        // Selección de la próxima generación
        population = select_new_generation(population, costs);
    }

    // Recolectar los mejores individuos de cada proceso
    Individual local_best_individual = population[0];
    for (const auto& ind : population) {
        if (ind.cost < local_best_individual.cost) {
            local_best_individual = ind;
        }
    }

    // Recolectar el mejor individuo global
    int local_best_cost = local_best_individual.cost;
    int global_best_cost;
    MPI_Allreduce(&local_best_cost, &global_best_cost, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Encontrar y comunicar el camino correspondiente al mejor costo global
    Individual global_best_individual;
    if (rank == 0) {
        global_best_individual.path.resize(NUM_CITIES);
    }

    if (local_best_cost == global_best_cost) {
        MPI_Gather(local_best_individual.path.data(), NUM_CITIES, MPI_INT,
                   global_best_individual.path.data(), NUM_CITIES, MPI_INT,
                   0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(NULL, NUM_CITIES, MPI_INT,
                   global_best_individual.path.data(), NUM_CITIES, MPI_INT,
                   0, MPI_COMM_WORLD);
    }

    // Medición del tiempo de ejecución
    if (rank == 0) {
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Imprimir resultados
        std::cout << "Mejor individuo (costo): " << global_best_cost << "\n";
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
