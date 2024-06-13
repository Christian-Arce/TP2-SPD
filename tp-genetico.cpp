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

struct Individual {
    int cost;
    std::vector<int> path;

    Individual() : cost(0), path(NUM_CITIES) {}
    Individual(int cost, std::vector<int> path) : cost(cost), path(path) {}
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

// Evaluación de aptitud para un individuo
int evaluar_Individuo(const std::vector<int>& path) {
    return evaluate_cost(path);
}

// Selección de individuos (torneo)
int seleccionar_Padres(const std::vector<int>& costs) {
    int tamTorneo = 3; // Tamaño del torneo
    int mejor = rand() % costs.size();
    for (int i = 1; i < tamTorneo; ++i) {
        int contendor = rand() % costs.size();
        if (costs[contendor] < costs[mejor]) {
            mejor = contendor;
        }
    }
    return mejor;
}

// Cruce de dos individuos utilizando cruce ordenado
std::vector<int> cruzar(const std::vector<int>& padre1, const std::vector<int>& padre2) {
    std::vector<int> hijo(NUM_CITIES, -1); // Inicializar con valores inválidos
    int inicio = rand() % NUM_CITIES;
    int fin = rand() % NUM_CITIES;
    if (inicio > fin) std::swap(inicio, fin);

    // Copiar segmento del padre1 al hijo
    for (int i = inicio; i <= fin; ++i) {
        hijo[i] = padre1[i];
    }

    // Copiar resto de ciudades del padre2
    int index = 0;
    for (int i = 0; i < NUM_CITIES; ++i) {
        if (hijo[i] == -1) {
            while (std::find(hijo.begin(), hijo.end(), padre2[index]) != hijo.end()) {
                ++index;
            }
            hijo[i] = padre2[index++];
        }
    }
    return hijo;
}

// Mutación mediante intercambio de dos ciudades
void mutar(std::vector<int>& path, double tasaMutacion) {
    for (int i = 0; i < NUM_CITIES; ++i) {
        if ((rand() / double(RAND_MAX)) < tasaMutacion) {
            int j = rand() % NUM_CITIES;
            std::swap(path[i], path[j]);
        }
    }
}

// Selección de la próxima generación
std::vector<Individual> seleccionar_Nueva_Generacion(const std::vector<Individual>& population, const std::vector<int>& costs, double tasaMutacion) {
    std::vector<Individual> nueva_generacion(POPULATION_SIZE);

    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE / 2; ++i) {
        int padre1 = seleccionar_Padres(costs);
        int padre2 = seleccionar_Padres(costs);

        std::vector<int> hijo1 = cruzar(population[padre1].path, population[padre2].path);
        std::vector<int> hijo2 = cruzar(population[padre2].path, population[padre1].path);

        mutar(hijo1, tasaMutacion);
        mutar(hijo2, tasaMutacion);

        nueva_generacion[2 * i] = Individual(evaluate_cost(hijo1), hijo1);
        nueva_generacion[2 * i + 1] = Individual(evaluate_cost(hijo2), hijo2);
    }

    return nueva_generacion;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank); // Semilla aleatoria única para cada proceso

    // Generar ciudades aleatorias en el proceso 0
    if (rank == 0) {
        generate_random_cities();
    }

    // Broadcast de las ciudades a todos los procesos
    MPI_Bcast(&cities[0][0], NUM_CITIES * NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

    // Parámetros del algoritmo genético
    const double tasaMutacion = 0.01;
    const int numGeneraciones = 100;

    // Inicialización de la población local en cada proceso
    std::vector<Individual> local_population(POPULATION_SIZE / size);

    // Inicialización de población en cada proceso
    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE / size; ++i) {
        std::vector<int> path(NUM_CITIES);
        for (int j = 0; j < NUM_CITIES; ++j) {
            path[j] = j;
        }
        std::random_shuffle(path.begin(), path.end()); // Mezclar aleatoriamente el camino inicial
        local_population[i] = Individual(evaluate_cost(path), path);
    }

    // Bucle principal del algoritmo genético
    for (int generacion = 0; generacion < numGeneraciones; ++generacion) {
        // Evaluación de aptitud en paralelo con OpenMP
        std::vector<int> costs(POPULATION_SIZE / size);
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE / size; ++i) {
            costs[i] = evaluar_Individuo(local_population[i].path);
        }

        // Reunir todos los costos en el proceso 0
        std::vector<int> all_costs(POPULATION_SIZE);
        MPI_Gather(costs.data(), POPULATION_SIZE / size, MPI_INT, all_costs.data(), POPULATION_SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

        // Selección de la próxima generación solo en el proceso 0
        if (rank == 0) {
            // Combine todas las poblaciones locales en una sola
            std::vector<Individual> all_population(POPULATION_SIZE);
            MPI_Gather(local_population.data(), POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, all_population.data(), POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, 0, MPI_COMM_WORLD);

            // Selección de la próxima generación
            all_population = seleccionar_Nueva_Generacion(all_population, all_costs, tasaMutacion);

            // Distribuir la nueva población entre los procesos
            for (int i = 0; i < size; ++i) {
                MPI_Scatter(all_population.data() + i * (POPULATION_SIZE / size), POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, local_population.data(), POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, i, MPI_COMM_WORLD);
            }
        } else {
            // Recibir la nueva población del proceso 0
            MPI_Scatter(nullptr, POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, local_population.data(), POPULATION_SIZE / size * sizeof(Individual), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }

    // Encontrar el mejor individuo local en cada proceso
    Individual local_best_individual = local_population[0];
    for (const auto& ind : local_population) {
        if (ind.cost < local_best_individual.cost) {
            local_best_individual = ind;
        }
    }

    // Recoger todos los mejores individuos locales en el proceso 0
    std::vector<int> all_best_costs(size);
    MPI_Gather(&local_best_individual.cost
