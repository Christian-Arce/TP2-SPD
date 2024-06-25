#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm> // Para std::shuffle
#include <random>    // Para std::default_random_engine

#define NUM_CITIES 100
#define POPULATION_SIZE 100  // Incrementar para mayor carga de trabajo
#define TOURNAMENT_SIZE 20
#define MUTATION_RATE 0.05
#define NUM_GENERATIONS 1000  // Número de generaciones

struct Individual {
    int cost;
    std::vector<int> path;

    Individual() : cost(0), path(NUM_CITIES) {}
    Individual(int cost, std::vector<int> path) : cost(cost), path(path) {}
};

std::vector<std::vector<int>> cities(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<Individual> population(POPULATION_SIZE);

// Función para generar ciudades aleatorias de manera determinista
void generate_random_cities(unsigned seed) {
    std::srand(seed);  // Usar la semilla proporcionada
    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            if (i != j) {
                cities[i][j] = std::rand() % 100 + 1;
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

// Función para inicializar la población de individuos con la misma semilla
void initialize_population(unsigned seed) {
    std::default_random_engine rng(seed);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        std::vector<int> path(NUM_CITIES);
        for (int j = 0; j < NUM_CITIES; ++j) {
            path[j] = j;
        }
        std::shuffle(path.begin(), path.end(), rng);
        population[i] = Individual(evaluate_cost(path), path);
    }
}

// Función para seleccionar una nueva generación de individuos usando torneo
Individual tournament_selection(const std::vector<Individual>& population) {
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, POPULATION_SIZE - 1);
    
    Individual best = population[dist(rng)];
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        std::uniform_int_distribution<int> local_dist(0, POPULATION_SIZE - 1);
        Individual contender = population[local_dist(rng)];
        #pragma omp critical
        {
            if (contender.cost < best.cost) {
                best = contender;
            }
        }
    }
    return best;
}

// Función para realizar el cruce (crossover)
Individual crossover(const Individual& parent1, const Individual& parent2) {
    std::vector<int> child_path(NUM_CITIES, -1);
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, NUM_CITIES - 1);
    
    int start = dist(rng);
    int end = dist(rng);
    
    if (start > end) std::swap(start, end);

    for (int i = start; i <= end; ++i) {
        child_path[i] = parent1.path[i];
    }

    int current_pos = (end + 1) % NUM_CITIES;
    for (int i = 0; i < NUM_CITIES; ++i) {
        int city = parent2.path[i];
        if (std::find(child_path.begin(), child_path.end(), city) == child_path.end()) {
            child_path[current_pos] = city;
            current_pos = (current_pos + 1) % NUM_CITIES;
        }
    }

    int cost = evaluate_cost(child_path);
    return Individual(cost, child_path);
}

// Función para realizar la mutación
void mutate(Individual& individual) {
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<double> mutation_chance(0.0, 1.0);
    std::uniform_int_distribution<int> dist(0, NUM_CITIES - 1);

    for (int i = 0; i < NUM_CITIES; ++i) {
        if (mutation_chance(rng) < MUTATION_RATE) {
            int j = dist(rng);
            std::swap(individual.path[i], individual.path[j]);
        }
    }
}

// Función para seleccionar una nueva generación de individuos
std::vector<Individual> select_new_generation(const std::vector<Individual>& population) {
    std::vector<Individual> new_population(POPULATION_SIZE);

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        Individual parent1 = tournament_selection(population);
        Individual parent2 = tournament_selection(population);
        Individual child = crossover(parent1, parent2);
        mutate(child);
        new_population[i] = child;
    }
    return new_population;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42);  // Semilla aleatoria basada en el rango para variar entre procesos

    auto start_time = std::chrono::steady_clock::now();

    if (rank == 0) {
        generate_random_cities(42);  // Generar ciudades aleatorias usando la misma semilla
    }

    // Broadcast de las ciudades generadas
    MPI_Bcast(&cities[0][0], NUM_CITIES * NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

    // Inicialización de la población con la misma semilla en todos los procesos
    initialize_population(42);

    // Ejemplo de cálculo de costos y selección
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        population = select_new_generation(population);
    }

    // Encontrar el mejor individuo local
    Individual local_best_individual = population[0];
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (population[i].cost < local_best_individual.cost) {
            //#pragma omp critical
            if (population[i].cost < local_best_individual.cost) {
                local_best_individual = population[i];
            }
        }
    }

    // Recoger todos los mejores individuos locales en el proceso 0
    std::vector<int> all_best_costs(size);
    MPI_Gather(&local_best_individual.cost, 1, MPI_INT, all_best_costs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> local_best_path = local_best_individual.path;
    std::vector<int> all_best_paths(size * NUM_CITIES);

    MPI_Gather(local_best_path.data(), NUM_CITIES, MPI_INT, all_best_paths.data(), NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);

    // Encontrar el mejor individuo global en el proceso 0
    Individual global_best_individual;
    if (rank == 0) {
        int min_cost = all_best_costs[0];
        int min_index = 0;

        for (int i = 1; i < size; ++i) {
            if (all_best_costs[i] < min_cost) {
                min_cost = all_best_costs[i];
                min_index = i;
            }
        }

        global_best_individual.cost = min_cost;
        global_best_individual.path.assign(all_best_paths.begin() + min_index * NUM_CITIES, all_best_paths.begin() + (min_index + 1) * NUM_CITIES);

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
