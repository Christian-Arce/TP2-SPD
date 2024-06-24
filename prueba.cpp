#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm> // Para std::shuffle
#include <random>    // Para std::default_random_engine
#include <climits>

#define NUM_CITIES 200
#define POPULATION_SIZE 100000  // Incrementar para mayor carga de trabajo

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

// Función para inicializar la población de individuos
void initialize_population(unsigned seed) {
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
    unsigned seed = 1098;  // Genera una semilla única basada en el tiempo actual
    srand(seed);  // Inicializa la semilla aleatoria

    auto start_time = std::chrono::steady_clock::now();

    generate_random_cities();

    // Inicialización de la población
    initialize_population(seed);

    // Ejemplo de cálculo de costos y selección
    std::vector<int> costs(POPULATION_SIZE);
    for (int generation = 0; generation < 1000; ++generation) {
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            costs[i] = evaluate_cost(population[i].path);
        }
        population = select_new_generation(population, costs);
    }

    // Encontrar el mejor individuo localmente
    Individual best_individual = population[0];
    #pragma omp parallel
    {
        Individual local_best_individual = population[0];
        #pragma omp for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            if (population[i].cost < local_best_individual.cost) {
                local_best_individual = population[i];
            }
        }
        
        #pragma omp critical
        {
            if (local_best_individual.cost < best_individual.cost) {
                best_individual = local_best_individual;
            }
        }
    }

    // Medición del tiempo de ejecución
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Imprimir resultados
    std::cout << "Mejor individuo (costo): " << best_individual.cost << "\n";
    std::cout << "Camino: ";
    for (int city : best_individual.path) {
        std::cout << city << " ";
    }
    std::cout << "\n";

    // Imprimir tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << elapsed_time << " milisegundos\n";

    return 0;
}
