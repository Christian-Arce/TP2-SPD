#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <mpi.h>

using namespace std;

struct City {
    int id;
    double x, y;
};

double calculateDistance(const City& city1, const City& city2) {
    double dx = city1.x - city2.x;
    double dy = city1.y - city2.y;
    return sqrt(dx * dx + dy * dy);
}

double calculateTotalDistance(const vector<int>& route, const vector<City>& cities) {
    double totalDistance = 0.0;
    int numCities = route.size();

    for (int i = 0; i < numCities - 1; ++i) {
        totalDistance += calculateDistance(cities[route[i]], cities[route[i + 1]]);
    }

    totalDistance += calculateDistance(cities[route[numCities - 1]], cities[route[0]]);

    return totalDistance;
}

std::vector<int> generateRandomRoute(int numCities) {
    std::vector<int> route(numCities);
    for (int i = 0; i < numCities; ++i) {
        route[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(route.begin(), route.end(), g);

    return route;
}

vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int numCities = parent1.size();
    vector<int> child(numCities, -1);

    int startPos = rand() % numCities;
    int endPos = rand() % numCities;

    if (startPos > endPos) {
        swap(startPos, endPos);
    }

    for (int i = startPos; i <= endPos; ++i) {
        child[i] = parent1[i];
    }

    int currentIndex = 0;
    for (int i = 0; i < numCities; ++i) {
        if (child[i] == -1) {
            while (find(child.begin(), child.end(), parent2[currentIndex]) != child.end()) {
                currentIndex = (currentIndex + 1) % numCities;
            }
            child[i] = parent2[currentIndex];
            currentIndex = (currentIndex + 1) % numCities;
        }
    }

    return child;
}

void mutate(vector<int>& route, double mutationRate) {
    int numCities = route.size();
    for (int i = 0; i < numCities; ++i) {
        if (rand() / static_cast<double>(RAND_MAX) < mutationRate) {
            int j = rand() % numCities;
            swap(route[i], route[j]);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string LIGHT_BLUE = "\033[94m";
    const string GREEN = "\033[32m";
    const string YELLOW = "\033[33m";
    const string RESET = "\033[0m";
    srand(42);
    if (rank == 0) {
        cout << GREEN + "================================================" + RESET << endl;
        cout << GREEN + "           TRAVELING SALESMAN PROBLEM     " + RESET << endl;
        cout << GREEN + "================================================" + RESET << endl << endl;
    }

    time_t startTime = time(nullptr);

    vector<City> cities;
    int numCities;

    if (rank == 0) {
        ifstream inputFile("/home/christian/tsp-test/Travelling-Salesman-Problem-Using-Genetic-Algorithm/pcb3038.tsp"); // Replace with the actual TSP file name
        if (!inputFile) {
            cerr << "Failed to open input file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;

        while (inputFile >> line) {
            if (line == "NODE_COORD_SECTION") {
                break;
            }
        }

        while (inputFile >> line) {
            if (line == "EOF") {
                break;
            }
            City city;
            inputFile >> city.id >> city.x >> city.y;
            cities.push_back(city);
        }

        numCities = cities.size();
    }

    MPI_Bcast(&numCities, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        cities.resize(numCities);
    }

    MPI_Bcast(cities.data(), numCities * sizeof(City), MPI_BYTE, 0, MPI_COMM_WORLD);

    int populationSize = 100;
    int numGenerations = 100;
    double mutationRate = 0.01;

    vector<int> bestRoute;
    double bestDistance = numeric_limits<double>::max();

    vector<vector<int>> population(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        population[i] = generateRandomRoute(numCities);
    }

    for (int generation = 0; generation < numGenerations; ++generation) {
        vector<double> fitness(populationSize);
        int routesPerProcess = populationSize / numProcesses;
        int start = rank * routesPerProcess;
        int end = (rank == numProcesses - 1) ? populationSize : start + routesPerProcess;

        for (int i = start; i < end; ++i) {
            fitness[i - start] = 1.0 / calculateTotalDistance(population[i], cities);
        }

        vector<double> allFitness(populationSize);
        MPI_Allgather(fitness.data(), routesPerProcess, MPI_DOUBLE, allFitness.data(), routesPerProcess, MPI_DOUBLE, MPI_COMM_WORLD);

        vector<pair<int, double>> fitnessPairs(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            fitnessPairs[i] = make_pair(i, allFitness[i]);
        }

        sort(fitnessPairs.begin(), fitnessPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second > b.second;
        });

        vector<vector<int>> newPopulation(populationSize);

        newPopulation[0] = population[fitnessPairs[0].first];
        if (1.0 / fitnessPairs[0].second < bestDistance) {
            bestRoute = population[fitnessPairs[0].first];
            bestDistance = 1.0 / fitnessPairs[0].second;
        }

        for (int i = 1; i < populationSize; ++i) {
            int parent1 = fitnessPairs[i - 1].first;
            int parent2 = fitnessPairs[i].first;
            newPopulation[i] = crossover(population[parent1], population[parent2]);
            mutate(newPopulation[i], mutationRate);
        }

        population = newPopulation;

        // Balance de carga dinámico: redistribuir trabajo si es necesario
        int totalRoutesComputed = populationSize;
        int averageRoutesPerProcess = totalRoutesComputed / numProcesses;
        int routesToReceive = averageRoutesPerProcess;

        if (rank == 0) {
            for (int p = 1; p < numProcesses; ++p) {
                int currentRoutes = (p == numProcesses - 1) ? totalRoutesComputed - (numProcesses - 1) * averageRoutesPerProcess : averageRoutesPerProcess;
                MPI_Send(&currentRoutes, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                for (int i = 0; i < currentRoutes; ++i) {
                    MPI_Send(population[p * averageRoutesPerProcess + i].data(), numCities, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Status status;
            MPI_Recv(&routesToReceive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < routesToReceive; ++i) {
                MPI_Recv(population[i].data(), numCities, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }
        }
    }

    vector<double> allBestDistances(numProcesses);
    MPI_Gather(&bestDistance, 1, MPI_DOUBLE, allBestDistances.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int globalBestIndex = distance(allBestDistances.begin(), min_element(allBestDistances.begin(), allBestDistances.end()));
        double globalBestDistance = allBestDistances[globalBestIndex];

        cout << YELLOW + "\nResults:" + RESET << endl;
        cout << LIGHT_BLUE + "Best route:\n";
        for (int city : bestRoute) {
            cout << city << " ";
        }
        cout << RESET << "\n\n";
        cout << GREEN << "Total distance: " << globalBestDistance << RESET << endl;

        time_t endTime = time(nullptr);
        double duration = difftime(endTime, startTime);
        cout << YELLOW + "Time taken by function: " << duration << " seconds" + RESET << endl;

        cout << GREEN + "\nThank you for using the Genetic Algorithm TSP Solver!" + RESET << endl;
        cout << GREEN + "===========================================" + RESET << endl;
    }

    MPI_Finalize();
    return 0;
}
