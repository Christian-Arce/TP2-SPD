#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <ctime> 
#include <random>
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
std::vector<int> generateRandomRoute(int numCities) {
    std::vector<int> route(numCities);
    for (int i = 0; i < numCities; ++i) {
        route[i] = i;
    }

    // Create a random device and random engine
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle the route using the random engine
    std::shuffle(route.begin(), route.end(), g);

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

int main() {
    srand(42);

    const string LIGHT_BLUE = "\033[94m";
    const string GREEN = "\033[32m";
    const string YELLOW = "\033[33m";
    const string RESET = "\033[0m";

    cout << GREEN + "================================================" + RESET << endl;
    cout << GREEN + "           TRAVELING SALESMAN PROBLEM     " + RESET << endl;
    cout << GREEN + "================================================" + RESET << endl << endl;

    // Record the start time
    time_t startTime = time(nullptr);

    //ifstream inputFile("pcb3038.tsp"); // Replace with the actual TSP file name
    ifstream inputFile("/home/christian/tsp-test/Travelling-Salesman-Problem-Using-Genetic-Algorithm/pcb3038.tsp"); // Replace with the actual TSP file name
    if (!inputFile) {
        cerr << "Failed to open input file." << endl;
        return 1;
    }

    string line;
    vector<City> cities;

    // Skip lines until the problem data begins
    while (inputFile >> line) {
        if (line == "NODE_COORD_SECTION") {
            break;
        }
    }

    // Read the coordinates from the TSP file
    while (inputFile >> line) {
        if (line == "EOF") {
            break;
        }
        City city;
        inputFile >> city.id >> city.x >> city.y;
        cities.push_back(city);
    }

    int numCities = cities.size();
    int populationSize = 100;
    int numGenerations = 100;
    double mutationRate = 0.01;

    vector<int> bestRoute;
    double bestDistance = numeric_limits<double>::max();

    // Initialize the population
    vector<vector<int>> population(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        population[i] = generateRandomRoute(numCities);
    }

    // Main Genetic Algorithm loop
    for (int generation = 0; generation < numGenerations; ++generation) {
        vector<pair<int, double>> fitness;
        for (int i = 0; i < populationSize; ++i) {
            double distance = calculateTotalDistance(population[i], cities);
            fitness.push_back(make_pair(i, 1.0 / distance));
        }

        sort(fitness.begin(), fitness.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second > b.second;
            });

        vector<vector<int>> newPopulation(populationSize);

        // Elitism: Keep the best route from the previous generation
        newPopulation[0] = population[fitness[0].first];
        if (1.0 / fitness[0].second < bestDistance) {
            bestRoute = population[fitness[0].first];
            bestDistance = 1.0 / fitness[0].second;
        }

        // Select parents and create offspring
        for (int i = 1; i < populationSize; ++i) {
            int parent1 = fitness[i - 1].first;
            int parent2 = fitness[i].first;
            newPopulation[i] = crossover(population[parent1], population[parent2]);
            mutate(newPopulation[i], mutationRate);
        }

        population = newPopulation;
    }


    cout << YELLOW + "\nResults:" + RESET << endl;
    cout << LIGHT_BLUE + "Best route:\n";
    for (int city : bestRoute) {
        cout << city << " ";
    }
    cout << RESET << "\n\n";
    cout << GREEN << "Total distance: " << bestDistance << RESET << endl;

    time_t endTime = time(nullptr);
    double duration = difftime(endTime, startTime);
    cout << YELLOW + "Time taken by function: " << duration << " seconds" + RESET << endl;

    cout << GREEN + "\nThank you for using the Genetic Algorithm TSP Solver!" + RESET << endl;
    cout << GREEN + "===========================================" + RESET << endl;

    return 0;
}
