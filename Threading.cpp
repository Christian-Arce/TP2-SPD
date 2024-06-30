#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>
#include <random>
#include <queue>
#include <condition_variable>


using namespace std;
using namespace chrono;

struct City {
    int id;
    double x, y;
};

// Function to calculate distance between two cities
double calculateDistance(const City& city1, const City& city2) {
    double dx = city1.x - city2.x;
    double dy = city1.y - city2.y;
    return sqrt(dx * dx + dy * dy);
}

// Function to calculate total distance of a route
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

// Function for crossover between two parent routes
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

// Function for mutation of a route
void mutate(vector<int>& route, double mutationRate) {
    int numCities = route.size();
    random_device rd;
    mt19937 g(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < numCities; ++i) {
        if (dist(g) < mutationRate) {
            int j = rand() % numCities;
            swap(route[i], route[j]);
        }
    }
}

mutex mtx;

// Define a function for the GA process
void geneticAlgorithm(vector<City>& cities, int populationSize, int numGenerations, double mutationRate, vector<int>& bestRoute, double& bestDistance, int threadId, queue<vector<int>>& taskQueue, mutex& taskMutex, condition_variable& workAvailable) {
    int numCities = cities.size();

    while (true) {
        vector<int> task;

        {
            unique_lock<mutex> lock(taskMutex);
            if (taskQueue.empty()) {
                return;
            }

            task = taskQueue.front();
            taskQueue.pop();
        }


        for (int generation = 0; generation < numGenerations; ++generation) {
            // Perform GA operations on the task

            // Crossover
            vector<int> child = crossover(task, generateRandomRoute(numCities));

            // Mutation
            mutate(child, mutationRate);

            // Update best route and distance if needed
            double childDistance = calculateTotalDistance(child, cities);
            mtx.lock();
            if (childDistance < bestDistance) {
                bestRoute = child;
                bestDistance = childDistance;
            }
            mtx.unlock();
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

    ifstream inputFile("/home/christian/tsp-test/Travelling-Salesman-Problem-Using-Genetic-Algorithm/pcb3038.tsp");
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
    int numThreads = 16; 

    vector<int> bestRoute;
    double bestDistance = numeric_limits<double>::max();

    cout << LIGHT_BLUE + "Total number of threads: " << numThreads << endl;

    // Create a work queue for tasks
    queue<vector<int>> taskQueue;
    mutex taskMutex;
    condition_variable workAvailable;

    // Divide the population into tasks and enqueue them
    vector<vector<int>> initialPopulation(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        initialPopulation[i] = generateRandomRoute(numCities);
    }

    // Enqueue tasks
    for (int i = 0; i < populationSize; ++i) {
        taskQueue.push(initialPopulation[i]);
    }

    auto startTime = high_resolution_clock::now();

    // Create and launch worker threads
    vector<thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(geneticAlgorithm, ref(cities), populationSize, 
            numGenerations, mutationRate, 
            ref(bestRoute), ref(bestDistance), i, 
            ref(taskQueue), ref(taskMutex), ref(workAvailable));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // End measuring time
    auto endTime = high_resolution_clock::now();

    // Calculate elapsed time in seconds
    auto duration = duration_cast<seconds>(endTime - startTime);
    cout << YELLOW + "\nResults:" + RESET << endl;
    cout << LIGHT_BLUE + "Best route:\n";
    for (int city : bestRoute) {
        cout << city << " ";
    }
    cout << endl;
    cout << RESET << "\n\n";
    cout << GREEN << "Total distance: " << bestDistance << endl;
    cout << YELLOW + "Time taken by function: " << duration.count() << " seconds" << endl;

    cout << GREEN + "\nThank you for using the Genetic Algorithm TSP Solver!" + RESET << endl;
    cout << GREEN + "===========================================" + RESET << endl;

    return 0;
}
