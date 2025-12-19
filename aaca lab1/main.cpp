#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <string>

#include <random>
#include <chrono>
#include <fstream>

const int INF = std::numeric_limits<int>::max();


struct HeapNode {
    int vertex;
    int distance;

    HeapNode(int v, int d) : vertex(v), distance(d) {}

    bool operator>(const HeapNode& other) const {
        return distance > other.distance;
    }
};


class TernaryHeap {
private:
    std::vector<HeapNode> heap;
    std::vector<int> position;

    int parent(int i) const { return (i - 1) / 3; }
    int leftChild(int i) const { return 3 * i + 1; }
    int middleChild(int i) const { return 3 * i + 2; }
    int rightChild(int i) const { return 3 * i + 3; }

    void heapifyUp(int i) {
        while (i > 0 && heap[i].distance < heap[parent(i)].distance) {
            std::swap(heap[i], heap[parent(i)]);
            position[heap[i].vertex] = i;
            position[heap[parent(i)].vertex] = parent(i);
            i = parent(i);
        }
    }

    void heapifyDown(int i) {
        int smallest = i;
        int l = leftChild(i);
        int m = middleChild(i);
        int r = rightChild(i);

        if (l < (int)heap.size() && heap[l].distance < heap[smallest].distance) smallest = l;
        if (m < (int)heap.size() && heap[m].distance < heap[smallest].distance) smallest = m;
        if (r < (int)heap.size() && heap[r].distance < heap[smallest].distance) smallest = r;

        if (smallest != i) {
            std::swap(heap[i], heap[smallest]);
            position[heap[i].vertex] = i;
            position[heap[smallest].vertex] = smallest;
            heapifyDown(smallest);
        }
    }

public:
    TernaryHeap(int n) : position(n, -1) {}

    bool empty() const { return heap.empty(); }
    bool contains(int vertex) const { return position[vertex] != -1; }
    size_t size() const { return heap.size(); }

    void push(int vertex, int distance) {
        heap.emplace_back(vertex, distance);
        position[vertex] = heap.size() - 1;
        heapifyUp(heap.size() - 1);
    }

    HeapNode extractMin() {
        if (heap.empty()) throw std::runtime_error("Heap is empty");

        HeapNode root = heap[0];
        position[root.vertex] = -1;

        if (heap.size() > 1) {
            heap[0] = heap.back();
            position[heap[0].vertex] = 0;
        }
        heap.pop_back();

        if (!heap.empty()) heapifyDown(0);

        return root;
    }

    void decreaseKey(int vertex, int newDistance) {
        int pos = position[vertex];
        if (pos == -1) return;
        if (heap[pos].distance <= newDistance) return;

        heap[pos].distance = newDistance;
        heapifyUp(pos);
    }
};


class Edge {
private:
    int source;
    int destination;
    int weight;

public:
    Edge(int u, int v, int w) : source(u), destination(v), weight(w) {}

    int getSource() const { return source; }
    int getDestination() const { return destination; }
    int getWeight() const { return weight; }

    std::string toString() const {
        return "(" + std::to_string(source) + " -> " +
            std::to_string(destination) + ", w=" +
            std::to_string(weight) + ")";
    }
};


class Graph {
private:
    int vertexCount;
    std::vector<std::vector<Edge>> adjacencyList;
    std::string graphName;

    void initializeSingleSource(std::vector<int>& distances,
        std::vector<int>& predecessors,
        int source) const {
        for (int i = 0; i < vertexCount; ++i) {
            distances[i] = INF;
            predecessors[i] = -1;
        }
        distances[source] = 0;
    }

    void relax(int u, const Edge& edge,
        std::vector<int>& distances,
        std::vector<int>& predecessors,
        TernaryHeap& heap) const {
        int v = edge.getDestination();
        int w = edge.getWeight();

        if (distances[u] != INF && distances[v] > distances[u] + w) {
            distances[v] = distances[u] + w;
            predecessors[v] = u;

            if (heap.contains(v))
                heap.decreaseKey(v, distances[v]);
            else
                heap.push(v, distances[v]);
        }
    }

    const std::vector<Edge>& getOutgoingEdges(int vertex) const {
        if (vertex >= 0 && vertex < vertexCount)
            return adjacencyList[vertex];
        static const std::vector<Edge> empty;
        return empty;
    }

public:


    Graph(int n, const std::string& name = "Graph")
        : vertexCount(n), adjacencyList(n), graphName(name) {
    }

    void addEdge(int u, int v, int weight) {
        if (u >= 0 && u < vertexCount && v >= 0 && v < vertexCount)
            adjacencyList[u].push_back(Edge(u, v, weight));
    }

    int getVertexCount() const { return vertexCount; }

    int getEdgeCount() const {
        int count = 0;
        for (const auto& list : adjacencyList) {
            count += list.size();
        }
        return count;
    }

    void clear() {
        for (auto& list : adjacencyList) {
            list.clear();
        }
    }

    std::pair<std::vector<int>, std::vector<int>> dijkstra(int source) const {
        std::vector<int> distances(vertexCount);
        std::vector<int> predecessors(vertexCount);
        initializeSingleSource(distances, predecessors, source);

        std::vector<bool> inS(vertexCount, false);
        TernaryHeap heap(vertexCount);

        heap.push(source, 0);

        while (!heap.empty()) {
            HeapNode u_node = heap.extractMin();
            int u = u_node.vertex;
            inS[u] = true;

            for (const Edge& edge : getOutgoingEdges(u)) {
                int v = edge.getDestination();
                if (!inS[v])
                    relax(u, edge, distances, predecessors, heap);
            }
        }

        return { distances, predecessors };
    }
    std::pair<std::vector<int>, std::vector<int>> labels(int source) const {
        std::vector<int> distances(vertexCount, INF);
        std::vector<int> predecessors(vertexCount, -1);
        std::vector<bool> visited(vertexCount, false);

        distances[source] = 0;

        for (int i = 0; i < vertexCount; ++i) {
            int u = -1;
            int minDist = INF;

            for (int v = 0; v < vertexCount; ++v) {
                if (!visited[v] && distances[v] < minDist) {
                    minDist = distances[v];
                    u = v;
                }
            }
            if (u == -1 || minDist == INF) {
                break;
            }

            visited[u] = true;

            for (const Edge& edge : getOutgoingEdges(u)) {
                int v = edge.getDestination();
                int weight = edge.getWeight();

                if (!visited[v] && distances[u] != INF &&
                    distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    predecessors[v] = u;
                }
            }
        }

        return { distances, predecessors };
    }

    std::vector<int> getPath(int target,
        const std::vector<int>& predecessors) const {
        std::vector<int> path;
        for (int v = target; v != -1; v = predecessors[v])
            path.push_back(v);
        std::reverse(path.begin(), path.end());
        return path;
    }

    void print() const {
        std::cout << "Graph \"" << graphName << "\" adjacency list:\n";
        for (int u = 0; u < vertexCount; ++u) {
            std::cout << "  " << u << ": ";
            for (const auto& edge : adjacencyList[u])
                std::cout << edge.toString() << " ";
            std::cout << "\n";
        }
    }
};

class GraphGenerator {
public:
    static void generateRandomGraph(Graph& graph, int m,
        std::mt19937& gen,
        std::uniform_int_distribution<int>& weightDist) {
        int n = graph.getVertexCount();

        int maxPossibleEdges = n * (n - 1);
        if (m > maxPossibleEdges) {
            m = maxPossibleEdges;
        }

        std::uniform_int_distribution<int> vertexDist(0, n - 1);
        std::vector<std::vector<bool>> edgeExists(n, std::vector<bool>(n, false));

        int edgesAdded = 0;
        int attempts = 0;
        const int maxAttempts = m * 10;

        while (edgesAdded < m && attempts < maxAttempts) {
            int u = vertexDist(gen);
            int v = vertexDist(gen);

            if (u != v && !edgeExists[u][v]) {
                int weight = weightDist(gen);
                graph.addEdge(u, v, weight);
                edgeExists[u][v] = true;
                edgesAdded++;
            }
            attempts++;
        }
    }
};

template<typename Func>
double measureTime(Func func, int iterations = 1) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

void runExperiment3_1() {
    std::cout << "\n=== Эксперимент 3.1: m ≈ n^2/10 и m ≈ n^2 ===\n";
    std::cout << "n от 1 до 10001 с шагом 100, q=1, r=10^6\n";
    std::cout << "Выполнение...\n";

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> weightDist(1, 1000000);

    std::ofstream file("experiment3_1.txt");
    file << "Эксперимент 3.1: m ≈ n^2/10 и m ≈ n^2\n";
    file << "n, m_sparse, time_dijkstra_sparse(ms), time_labels_sparse(ms), "
        << "m_dense, time_dijkstra_dense(ms), time_labels_dense(ms)\n";
    file << "==============================================================\n";

    for (int n = 1; n <= 10001; n += 100) {
        std::cout << "n = " << n << " ... ";

        Graph graphSparse(n);
        int target_m_sparse = (n * n) / 10;
        if (target_m_sparse < 100) target_m_sparse = 100;

        GraphGenerator::generateRandomGraph(graphSparse, target_m_sparse, gen, weightDist);
        int actual_m_sparse = graphSparse.getEdgeCount();

        double time_dijkstra_sparse = measureTime([&]() {
            graphSparse.dijkstra(0);
            });

        double time_labels_sparse = measureTime([&]() {
            graphSparse.labels(0);
            });

        std::cout << "sparse: D=" << time_dijkstra_sparse << "ms, L=" << time_labels_sparse << "ms ... ";

        Graph graphDense(n);
        int target_m_dense = n * (n - 1);
        if (target_m_dense > 1000000) target_m_dense = 1000000;

        GraphGenerator::generateRandomGraph(graphDense, target_m_dense, gen, weightDist);
        int actual_m_dense = graphDense.getEdgeCount();

        double time_dijkstra_dense = measureTime([&]() {
            graphDense.dijkstra(0);
            });

        double time_labels_dense = measureTime([&]() {
            graphDense.labels(0);
            });

        std::cout << "dense: D=" << time_dijkstra_dense << "ms, L=" << time_labels_dense << "ms\n";

        file << n << "\t"
            << actual_m_sparse << "\t" << time_dijkstra_sparse << "\t" << time_labels_sparse << "\t"
            << actual_m_dense << "\t" << time_dijkstra_dense << "\t" << time_labels_dense << "\n";
    }

    file.close();
    std::cout << "✓ Эксперимент 3.1 завершен. Данные в experiment3_1.txt\n";
}

void runExperiment3_2() {
    std::cout << "\n=== Эксперимент 3.2: m ≈ 100*n и m ≈ 1000*n ===\n";
    std::cout << "n от 101 до 10001 с шагом 100, q=1, r=10^6\n";
    std::cout << "Выполнение...\n";

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> weightDist(1, 1000000);

    std::ofstream file("experiment3_2.txt");
    file << "Эксперимент 3.2: m ≈ 100*n и m ≈ 1000*n\n";
    file << "n, m_100, time_dijkstra_100(ms), time_labels_100(ms), "
        << "m_1000, time_dijkstra_1000(ms), time_labels_1000(ms)\n";
    file << "==============================================================\n";

    for (int n = 101; n <= 10001; n += 100) {
        std::cout << "n = " << n << " ... ";

        Graph graph100(n);
        int target_m_100 = 100 * n;

        GraphGenerator::generateRandomGraph(graph100, target_m_100, gen, weightDist);
        int actual_m_100 = graph100.getEdgeCount();

        double time_dijkstra_100 = measureTime([&]() {
            graph100.dijkstra(0);
            });

        double time_labels_100 = measureTime([&]() {
            graph100.labels(0);
            });

        std::cout << "m100: D=" << time_dijkstra_100 << "ms, L=" << time_labels_100 << "ms ... ";

        Graph graph1000(n);
        int target_m_1000 = 1000 * n;

        GraphGenerator::generateRandomGraph(graph1000, target_m_1000, gen, weightDist);
        int actual_m_1000 = graph1000.getEdgeCount();

        double time_dijkstra_1000 = measureTime([&]() {
            graph1000.dijkstra(0);
            });

        double time_labels_1000 = measureTime([&]() {
            graph1000.labels(0);
            });

        std::cout << "m1000: D=" << time_dijkstra_1000 << "ms, L=" << time_labels_1000 << "ms\n";

        file << n << "\t"
            << actual_m_100 << "\t" << time_dijkstra_100 << "\t" << time_labels_100 << "\t"
            << actual_m_1000 << "\t" << time_dijkstra_1000 << "\t" << time_labels_1000 << "\n";
    }

    file.close();
    std::cout << "✓ Эксперимент 3.2 завершен. Данные в experiment3_2.txt\n";
}

void runExperiment3_3() {
    std::cout << "\n=== Эксперимент 3.3: n=10001, m=0..10^7 ===\n";
    std::cout << "n = 10001, m от 0 до 10,000,000 с шагом 1000, q=1, r=10^6\n";
    std::cout << "Выполнение...\n";

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> weightDist(1, 1000000);

    std::ofstream file("experiment3_3.txt");
    file << "Эксперимент 3.3: n=10001, m=0..10^7\n";
    file << "m, time_dijkstra(ms), time_labels(ms)\n";
    file << "=====================================\n";

    const int n = 10001;
    const int max_m = 10000000;
    const int step = 1000;

    Graph graph(n);

    for (int m = 0; m <= max_m; m += step) {
        if (m % 100000 == 0) {
            std::cout << "m = " << m << " ... ";
        }

        GraphGenerator::generateRandomGraph(graph, m, gen, weightDist);
        int actual_m = graph.getEdgeCount();

        double time_dijkstra = measureTime([&]() {
            graph.dijkstra(0);
            });

        double time_labels = measureTime([&]() {
            graph.labels(0);
            });

        if (m % 100000 == 0) {
            std::cout << "D=" << time_dijkstra << "ms, L=" << time_labels << "ms\n";
        }

        file << actual_m << "\t" << time_dijkstra << "\t" << time_labels << "\n";

        graph.clear();
    }

    file.close();
    std::cout << "✓ Эксперимент 3.3 завершен. Данные в experiment3_3.txt\n";
}

void runExperiment3_4() {
    std::cout << "\n=== Эксперимент 3.4: n=10001, r=1..200 ===\n";
    std::cout << "n = 10001, q=1, r от 1 до 200\n";
    std::cout << "Выполнение...\n";

    std::mt19937 gen(42);

    std::ofstream file("experiment3_4.txt");
    file << "Эксперимент 3.4: n=10001, r=1..200\n";
    file << "r, time_dijkstra_dense(ms), time_labels_dense(ms), "
        << "time_dijkstra_sparse(ms), time_labels_sparse(ms)\n";
    file << "==============================================================\n";

    const int n = 10001;

    for (int r = 1; r <= 200; r++) {
        if (r % 20 == 0) {
            std::cout << "r = " << r << " ... ";
        }

        std::uniform_int_distribution<int> weightDist(1, r);

        Graph graphDense(n);
        int target_m_dense = (n * n) / 10;
        if (target_m_dense > 1000000) target_m_dense = 1000000;

        GraphGenerator::generateRandomGraph(graphDense, target_m_dense, gen, weightDist);

        double time_dijkstra_dense = measureTime([&]() {
            graphDense.dijkstra(0);
            });

        double time_labels_dense = measureTime([&]() {
            graphDense.labels(0);
            });

        Graph graphSparse(n);
        int target_m_sparse = 1000 * n;

        GraphGenerator::generateRandomGraph(graphSparse, target_m_sparse, gen, weightDist);

        double time_dijkstra_sparse = measureTime([&]() {
            graphSparse.dijkstra(0);
            });

        double time_labels_sparse = measureTime([&]() {
            graphSparse.labels(0);
            });

        if (r % 20 == 0) {
            std::cout << "dense: D=" << time_dijkstra_dense << "ms, L=" << time_labels_dense
                << "ms | sparse: D=" << time_dijkstra_sparse << "ms, L=" << time_labels_sparse << "ms\n";
        }

        file << r << "\t"
            << time_dijkstra_dense << "\t" << time_labels_dense << "\t"
            << time_dijkstra_sparse << "\t" << time_labels_sparse << "\n";
    }

    file.close();
    std::cout << "✓ Эксперимент 3.4 завершен. Данные в experiment3_4.txt\n";
}

int main() {
    std::cout << "=== ЗАПУСК ЭКСПЕРИМЕНТОВ ===\n";

    //runExperiment3_1();
    //runExperiment3_2();
    //runExperiment3_3();
    runExperiment3_4();

    return 0;
}



//int main() {
//    Graph g(6, "Test Graph");
//
//    g.addEdge(0, 1, 7);
//    g.addEdge(0, 2, 9);
//    g.addEdge(0, 5, 14);
//    g.addEdge(1, 2, 10);
//    g.addEdge(1, 3, 15);
//    g.addEdge(2, 3, 11);
//    g.addEdge(2, 5, 2);
//    g.addEdge(3, 4, 6);
//    g.addEdge(5, 4, 9);
//
//    g.print();
//    std::cout << "\n---------------------\n";
//
//    int source = 0;
//
//    auto result = g.dijkstra(source);
//    std::vector<int> distances = result.first;
//    std::vector<int> predecessors = result.second;
//
//    std::cout << "\nShortest distances from vertex " << source << " (Dijkstra):\n";
//    for (int i = 0; i < g.getVertexCount(); ++i) {
//        std::cout << "  to " << i << " = ";
//        if (distances[i] == INF)
//            std::cout << "INF";
//        else
//            std::cout << distances[i];
//        std::cout << "\n";
//    }
//
//    std::cout << "\nPaths:\n";
//    for (int i = 0; i < g.getVertexCount(); ++i) {
//        auto path = g.getPath(i, predecessors);
//        std::cout << "  " << source << " -> " << i << ": ";
//        for (size_t j = 0; j < path.size(); ++j) {
//            std::cout << path[j];
//            if (j + 1 < path.size()) std::cout << " -> ";
//        }
//        std::cout << "\n";
//    }
//    std::cout << "---------------------\n";
//
//    std::cout << "\n Shortest distances from vertex " << source << " (Labels):\n";
//    auto result2 = g.labels(source);
//    std::vector<int> distances2 = result2.first;
//    std::vector<int> predecessors2 = result2.second;
//
//    std::cout << "Shortest distances from vertex " << source << ":\n";
//    for (int i = 0; i < g.getVertexCount(); ++i) {
//        std::cout << "  to " << i << " = ";
//        if (distances2[i] == INF) std::cout << "INF";
//        else std::cout << distances2[i];
//        std::cout << "\n";
//    }
//    std::cout << "\nPaths (from Labels algorithm):\n";
//    for (int i = 0; i < g.getVertexCount(); ++i) {
//        auto path = g.getPath(i, predecessors2);
//        std::cout << "  " << source << " -> " << i << ": ";
//        for (size_t j = 0; j < path.size(); ++j) {
//            std::cout << path[j];
//            if (j + 1 < path.size()) std::cout << " -> ";
//        }
//        std::cout << "\n";
//    }
//
//
//    return 0;
//}