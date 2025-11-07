#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <string>

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


int main() {
    Graph g(6, "Test Graph");

    g.addEdge(0, 1, 7);
    g.addEdge(0, 2, 9);
    g.addEdge(0, 5, 14);
    g.addEdge(1, 2, 10);
    g.addEdge(1, 3, 15);
    g.addEdge(2, 3, 11);
    g.addEdge(2, 5, 2);
    g.addEdge(3, 4, 6);
    g.addEdge(5, 4, 9);

    g.print();

    int source = 0;

    auto result = g.dijkstra(source);
    std::vector<int> distances = result.first;
    std::vector<int> predecessors = result.second;

    std::cout << "\nShortest distances from vertex " << source << ":\n";
    for (int i = 0; i < g.getVertexCount(); ++i) {
        std::cout << "  to " << i << " = ";
        if (distances[i] == INF)
            std::cout << "INF";
        else
            std::cout << distances[i];
        std::cout << "\n";
    }

    std::cout << "\nPaths:\n";
    for (int i = 0; i < g.getVertexCount(); ++i) {
        auto path = g.getPath(i, predecessors);
        std::cout << "  " << source << " -> " << i << ": ";
        for (size_t j = 0; j < path.size(); ++j) {
            std::cout << path[j];
            if (j + 1 < path.size()) std::cout << " -> ";
        }
        std::cout << "\n";
    }

    return 0;
}