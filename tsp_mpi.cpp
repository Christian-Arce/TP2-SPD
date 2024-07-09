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

// Function to calculate the distance between two cities
double calculateDistance(const City& city1, const City& city2) {
    double dx = city1.x - city2.x;
    double dy = city1.y - city2.y;
    return sqrt(dx * dx + dy * dy);
}

// Function to calculate the total distance of a route
double calculateTotalDistance(const vector<int>& route, const vector<City>& cities) {
    double totalDistance = 0.0;
    int numCities = route.size();

    for (int i = 0; i < numCities - 1; ++i) {
        totalDistance += calculateDistance(cities[route[i]], cities[route[i + 1]]);
    }

    totalDistance += calculateDistance(cities[route[numCities - 1]], cities[route[0]]);

    return totalDistance;
}

// Function to generate a random route
vector<int> generateRandomRoute(int numCities) {
    vector<int> route(numCities);
    for (int i = 0; i < numCities; ++i) {
        route[i] = i;
    }

    random_device rd;
    mt19937 g(rd());

    shuffle(route.begin(), route.end(), g);

    return route;
}

// Crossover function between two parent routes
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

// Mutation function for a route
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

    vector<City> cities;
    int numCities;

    // Process 0 reads the input data and broadcasts to other processes
    if (rank == 0) {
        ifstream inputFile("/home/christian/tsp-test/Travelling-Salesman-Problem-Using-Genetic-Algorithm/pcb3038.tsp"); // Replace with your TSP file name
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

    // Broadcast the number of cities to all processes
    MPI_Bcast(&numCities, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the cities vector for non-root processes
    if (rank != 0) {
        cities.resize(numCities);
    }

    // Broadcast the cities data to all processes
    MPI_Bcast(cities.data(), numCities * sizeof(City), MPI_BYTE, 0, MPI_COMM_WORLD);

    int populationSize = 500;
    int numGenerations = 100;
    double mutationRate = 0.01;

    vector<int> localBestRoute(numCities);
    double localBestDistance = numeric_limits<double>::max();

    int routesPerProcess = populationSize / numProcesses;
    vector<vector<int>> population(routesPerProcess, vector<int>(numCities));

    // Initialize the local population for each process
    for (int i = 0; i < routesPerProcess; ++i) {
        population[i] = generateRandomRoute(numCities);
    }

    // Initialize start time
    time_t startTime = time(nullptr);
    
    for (int generation = 1; generation <= numGenerations; generation++) {
        //vector<double> fitness(routesPerProcess);
        // Evaluate fitness for local population
        vector<pair<int, double>> fitnessPairs;
        for (int i = 0; i < routesPerProcess; ++i) {
            double distance = calculateTotalDistance(population[i], cities);
            fitnessPairs.push_back(make_pair(i, 1.0 / distance));
        }

        // Sort based on fitness
        sort(fitnessPairs.begin(), fitnessPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second > b.second;
        });

        // New population for the next generation
        vector<vector<int>> newPopulation(routesPerProcess, vector<int>(numCities));

        // Keep the best individual unchanged
        newPopulation[0] = population[fitnessPairs[0].first];

        // Update global best route if a new local best is found
        if (1.0 / fitnessPairs[0].second < localBestDistance) {
            localBestRoute = population[fitnessPairs[0].first];
            localBestDistance = 1.0 / fitnessPairs[0].second;
        }

        // Periodically share the best route with all processes
        if (generation % 10 == 0) {
            double globalBestDistance;
            int senderRank = -1;
            MPI_Allreduce(&localBestDistance, &globalBestDistance, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

            printf("Rank %d: localBestDistance = %f, globalBestDistance = %f, generation = %d\n", rank, localBestDistance, globalBestDistance, generation);

            // Determine senderRank based on the process with the best global distance
            if (localBestDistance == globalBestDistance) {
                senderRank = rank;
            }

            // Ensure only one process has the senderRank as its rank
            MPI_Allreduce(MPI_IN_PLACE, &senderRank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            // Broadcast the best global route to all processes if senderRank is valid
            printf("Rank %d: Broadcasting from senderRank %d\n", rank, senderRank);

            // Broadcast the best global route to all processes
            MPI_Bcast(localBestRoute.data(), numCities, MPI_INT, senderRank, MPI_COMM_WORLD);
           
            // Declare a variable to store the combined result of localBestRoute
            vector<int> combinedBestRoute(numCities);

            // Perform a reduction to combine localBestRoute from all processes
            MPI_Reduce(localBestRoute.data(), combinedBestRoute.data(), numCities, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

            // Check if all processes have the same combined route
            if (rank == 0) {
                bool consistent = true;
                for (int i = 1; i < numProcesses; ++i) {
                    // Compare combinedBestRoute of all processes with process 0
                    if (memcmp(combinedBestRoute.data(), localBestRoute.data(), numCities * sizeof(int)) != 0) {
                        consistent = false;
                        break;
                    }
                }
    
                if (consistent) {
                    printf("MPI_Reduce successful. All processes have the same localBestRoute.\n");
                } else {
                    printf("MPI_Reduce failed. Data inconsistency detected.\n");
                }
            }
   
            for (int i = 0; i < routesPerProcess; ++i) {
                int parent1 = fitnessPairs[i - 1].first;  
                int parent2 = fitnessPairs[i].first;
                // Use the best global route if not process 0
                if (rank != 0) {
                    newPopulation[i] = crossover(localBestRoute, population[parent2]);
                } else {
                    newPopulation[i] = crossover(population[parent1], population[parent2]);
                }

                mutate(newPopulation[i], mutationRate);
            }

            // Update the local population for the next generation
            population = newPopulation;
        } else {
            // Apply crossover and mutation to generate new individuals
            for (int i = 0; i < routesPerProcess; ++i) {
                int parent1 = fitnessPairs[i - 1].first;  
                int parent2 = fitnessPairs[i].first;
                newPopulation[i] = crossover(population[parent1], population[parent2]);
                mutate(newPopulation[i], mutationRate);
            }
            // Update the local population for the next generation
            population = newPopulation;
        }
    }

    // Gather all best distances from all processes
    vector<double> allBestDistances(numProcesses);
    MPI_Gather(&localBestDistance, 1, MPI_DOUBLE, allBestDistances.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // After MPI_Gather of the best distances, the root process prints the results
    if (rank == 0) {
        // Find the index of the best global distance
        int globalBestIndex = distance(allBestDistances.begin(), min_element(allBestDistances.begin(), allBestDistances.end()));
        double globalBestDistance = allBestDistances[globalBestIndex];

        // Print the best route and its total distance
        cout << YELLOW + "\nResults:" + RESET << endl;
        cout << LIGHT_BLUE + "Best route:\n";
        for (int city : localBestRoute) {
            cout << city << " ";
        }
        cout << RESET << "\n\n";
        cout << GREEN << "Total distance: " << globalBestDistance << RESET << endl;

        // Calculate and display the execution time
        time_t endTime = time(nullptr);
        double duration = difftime(endTime, startTime);
        cout << YELLOW + "Time taken by function: " << duration << " seconds" + RESET << endl;

        cout << GREEN + "\nThank you for using the Genetic Algorithm TSP Solver!" + RESET << endl;
        cout << GREEN + "===========================================" + RESET << endl;
    }

    MPI_Finalize();
    return 0;
}
