#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <ctime>
#include <chrono>

// Función de aptitud: x^2
double get_Aptitude(const std::vector<int>& individuo) {
    int x = 0;
    for (int i = individuo.size() - 1, j = 0; i >= 0; --i, ++j) {
        x += individuo[i] * std::pow(2, j);
    }
    return std::pow(x, 2);
}

// Evaluación de aptitud para un individuo
double evaluar_Individuo(const std::vector<int>& individuo) {
    return get_Aptitude(individuo);
}

// Selección de individuos (torneo)
int seleccionar_Padres(const std::vector<double>& aptitudes) {
    int tamTorneo = 3; // Tamaño del torneo
    int mejor = rand() % aptitudes.size();
    for (int i = 1; i < tamTorneo; ++i) {
        int contendor = rand() % aptitudes.size();
        if (aptitudes[contendor] > aptitudes[mejor]) {
            mejor = contendor;
        }
    }
    return mejor;
}

// Cruce de un punto
std::vector<int> cruzar(const std::vector<int>& padre1, const std::vector<int>& padre2) {
    int puntoCruce = rand() % padre1.size();
    std::vector<int> hijo(padre1.size());
    for (int i = 0; i < padre1.size(); ++i) {
        if (i < puntoCruce) {
            hijo[i] = padre1[i];
        } else {
            hijo[i] = padre2[i];
        }
    }
    return hijo;
}

// Mutación
void mutar(std::vector<int>& individuo, double tasaMutacion) {
    for (int i = 0; i < individuo.size(); ++i) {
        if ((rand() / double(RAND_MAX)) < tasaMutacion) {
            individuo[i] = !individuo[i];
        }
    }
}

// Selección de la próxima generación
std::vector<std::vector<int>> seleccionar_Nueva_Generacion(const std::vector<std::vector<int>>& poblacion, const std::vector<double>& aptitudes, double tasaMutacion) {
    std::vector<std::vector<int>> nueva_generacion(poblacion.size(), std::vector<int>(poblacion[0].size()));
    #pragma omp parallel for
    for (size_t i = 0; i < poblacion.size() / 2; ++i) {
        int padre1 = seleccionar_Padres(aptitudes);
        int padre2 = seleccionar_Padres(aptitudes);

        // Cruzar
        std::vector<int> hijo1 = cruzar(poblacion[padre1], poblacion[padre2]);
        std::vector<int> hijo2 = cruzar(poblacion[padre2], poblacion[padre1]);

        // Mutar
        mutar(hijo1, tasaMutacion);
        mutar(hijo2, tasaMutacion);

        // Agregar hijos a la nueva generación
        nueva_generacion[2 * i] = hijo1;
        nueva_generacion[2 * i + 1] = hijo2;
    }
    return nueva_generacion;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(0) + rank); // Semilla aleatoria única para cada proceso

    // Parámetros del algoritmo genético
    const int tamPoblacion = 8000;
    const int tamanoIndividuo = 18;
    const int numGeneraciones = 12000;
    const double tasaMutacion = 0.01;

    // Calcular el tamaño de la población local para cada proceso
    int tamPoblacionLocal = tamPoblacion / size;

    // Inicialización de población local en cada proceso
    std::vector<std::vector<int>> poblacion(tamPoblacionLocal, std::vector<int>(tamanoIndividuo));
    #pragma omp parallel for
    for (int i = 0; i < tamPoblacionLocal; ++i) {
        for (int j = 0; j < tamanoIndividuo; ++j) {
            poblacion[i][j] = rand() % 2; // Genera aleatoriamente 0 o 1
        }
    }

    // Almacenar las aptitudes de los individuos
    std::vector<double> aptitudes(tamPoblacionLocal);

    // Medición de tiempo de ejecución
    auto start_time = std::chrono::steady_clock::now();

    // Bucle principal del algoritmo genético
    for (int generacion = 0; generacion < numGeneraciones; ++generacion) {
        // Evaluación de aptitud en paralelo con OpenMP
        #pragma omp parallel for
        for (int i = 0; i < tamPoblacionLocal; ++i) {
            aptitudes[i] = evaluar_Individuo(poblacion[i]);
        }

        // Reunir todas las aptitudes en el proceso 0
        std::vector<double> all_aptitudes(tamPoblacion);
        MPI_Gather(aptitudes.data(), tamPoblacionLocal, MPI_DOUBLE, all_aptitudes.data(), tamPoblacionLocal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Selección de la próxima generación solo en el proceso 0
        if (rank == 0) {
            // Combine todas las poblaciones locales en una sola
            std::vector<std::vector<int>> all_poblacion(tamPoblacion);
            std::vector<std::vector<int>> nueva_poblacion(tamPoblacion);
            MPI_Gather(poblacion.data(), tamPoblacionLocal * tamanoIndividuo, MPI_INT, all_poblacion.data(), tamPoblacionLocal * tamanoIndividuo, MPI_INT, 0, MPI_COMM_WORLD);

            // Convertir la población recibida en un vector plano a un vector de vectores
            for (int i = 0; i < tamPoblacion; ++i) {
                nueva_poblacion[i].resize(tamanoIndividuo);
                for (int j = 0; j < tamanoIndividuo; ++j) {
                    nueva_poblacion[i][j] = all_poblacion[i][j];
                }
            }

            // Selección de la próxima generación
            nueva_poblacion = seleccionar_Nueva_Generacion(nueva_poblacion, all_aptitudes, tasaMutacion);

            // Distribuir la nueva población entre los procesos
            for (int i = 0; i < size; ++i) {
                MPI_Scatter(nueva_poblacion.data() + i * tamPoblacionLocal, tamPoblacionLocal * tamanoIndividuo, MPI_INT, poblacion.data(), tamPoblacionLocal * tamanoIndividuo, MPI_INT, i, MPI_COMM_WORLD);
            }
        } else {
            // Recibir la nueva población del proceso 0
            MPI_Scatter(nullptr, tamPoblacionLocal * tamanoIndividuo, MPI_INT, poblacion.data(), tamPoblacionLocal * tamanoIndividuo, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    // Medición de tiempo de ejecución
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Encontrar el individuo con mayor aptitud en cada proceso
    double mejor_aptitud = aptitudes[0];
    int mejor_individuo = 0;
    for (size_t i = 1; i < aptitudes.size(); ++i) {
        if (aptitudes[i] > mejor_aptitud) {
            mejor_aptitud = aptitudes[i];
            mejor_individuo = i;
        }
    }

    // Reunir las mejores aptitudes y los mejores individuos en el proceso 0
    std::vector<double> all_mejor_aptitudes(size);
    std::vector<int> all_mejor_individuos(size * tamanoIndividuo);
    MPI_Gather(&mejor_aptitud, 1, MPI_DOUBLE, all_mejor_aptitudes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(poblacion[mejor_individuo].data(), tamanoIndividuo, MPI_INT, all_mejor_individuos.data(), tamanoIndividuo, MPI_INT, 0, MPI_COMM_WORLD);

    // Proceso 0 encuentra el mejor individuo global
    if (rank == 0) {
        double global_mejor_aptitud = all_mejor_aptitudes[0];
        int global_mejor_individuo = 0;
        for (int i = 1; i < size; ++i) {
            if (all_mejor_aptitudes[i] > global_mejor_aptitud) {
                global_mejor_aptitud = all_mejor_aptitudes[i];
                global_mejor_individuo = i;
            }
        }

        // Imprimir resultados
        std::cout << "Mejor individuo: ";
        for (int bit : all_mejor_individuos) {
            std::cout << bit;
        }
        std::cout << " con aptitud: " << global_mejor_aptitud << std::endl;

        // Imprimir tiempo de ejecución
        std::cout << "Tiempo de ejecución: " << elapsed_time << " milisegundos" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
