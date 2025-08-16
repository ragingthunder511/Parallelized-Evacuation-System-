#include <iostream>
#include <fstream>
#include <climits>
#include <vector>
#include <queue>
#include <algorithm>
#include <cuda.h>

#define MAX_CITIES 10000
#define MAX_EDGES 2000000
#define MAX_PATH 1000
#define MAX_DROPS 100

using namespace std;

struct Edge {
    int to;
    int length;
};

// Graph representation
__device__ __host__ inline int idx(int u, int v, int num_cities) {
    return u * num_cities + v;
}

__device__ void dijkstra(int src, int* edge_tos, int* edge_lens, int* edge_next, int* head,
                         int num_cities, int* dist, int* parent) {
    bool visited[MAX_CITIES] = {false};
    for (int i = 0; i < num_cities; ++i) {
        dist[i] = INT_MAX;
        parent[i] = -1;
    }
    dist[src] = 0;

    for (int count = 0; count < num_cities; ++count) {
        int u = -1;
        for (int i = 0; i < num_cities; ++i) {
            if (!visited[i] && (u == -1 || dist[i] < dist[u]))
                u = i;
        }

        if (u == -1 || dist[u] == INT_MAX) break;
        visited[u] = true;

        for (int e = head[u]; e != -1; e = edge_next[e]) {
            int v = edge_tos[e];
            int len = edge_lens[e];
            if (dist[u] + len < dist[v]) {
                dist[v] = dist[u] + len;
                parent[v] = u;
            }
        }
    }
}

__device__ int find_best_shelter(int* dist, int* is_shelter, int* shelter_capacity, int num_cities) {
    int min_dist = INT_MAX;
    int best_shelter = -1;
    for (int s = 0; s < num_cities; ++s) {
        if (is_shelter[s] && dist[s] < min_dist && shelter_capacity[s] > 0) {
            min_dist = dist[s];
            best_shelter = s;
        }
    }
    return best_shelter;
}

__global__ void evacuate_kernel(int num_populated, int* pop_city, int* population_prime, int* population_elder,
                                 int* edge_heads, int* edge_tos, int* edge_lens, int* edge_next, int* head,
                                 int num_cities, int* is_shelter, int* shelter_capacity, int max_dist_elderly,
                                 long long* path_size, long long* paths, long long* num_drops, long long* drops) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_populated) return;

    int src = pop_city[i];
    int remaining_prime = population_prime[i];
    int remaining_elder = population_elder[i];

    int drop_idx = 0;
    bool first_path = true;

    while (remaining_prime + remaining_elder > 0) {
        int dist[MAX_CITIES];
        int parent[MAX_CITIES];
        dijkstra(src, edge_tos, edge_lens, edge_next, head, num_cities, dist, parent);

        int best_shelter = find_best_shelter(dist, is_shelter, shelter_capacity, num_cities);
        if (best_shelter == -1) {
            if (remaining_prime + remaining_elder > 0) {
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 0] = src;
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 1] = remaining_prime;
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 2] = remaining_elder;
                drop_idx++;
            }
            break;
        }

        // Reconstruct path
        int temp_path[MAX_PATH];
        int path_len = 0;
        int node = best_shelter;
        while (node != -1 && path_len < MAX_PATH) {
            temp_path[path_len++] = node;
            if (node == src) break;
            node = parent[node];
        }

        if (node != src) break;

        // Reverse path
        for (int j = 0; j < path_len / 2; ++j) {
            int tmp = temp_path[j];
            temp_path[j] = temp_path[path_len - 1 - j];
            temp_path[path_len - 1 - j] = tmp;
        }

        // Check for mid-path elderly drop
        int total_dist = 0;
        int last_city = temp_path[0];
        bool dropped_midway = false;
        int cutoff_path_len = path_len;

        for (int j = 1; j < path_len; ++j) {
            int curr_city = temp_path[j];
            int edge_len = -1;

            // Find edge length
            for (int e = head[last_city]; e != -1; e = edge_next[e]) {
                if (edge_tos[e] == curr_city) {
                    edge_len = edge_lens[e];
                    break;
                }
            }
            if (edge_len == -1) break; // edge not found

            // 🔒 PREVENT exceeding max elderly range
            if (total_dist + edge_len > max_dist_elderly && remaining_elder > 0) {
                // ✅ drop elderly at last_city — BEFORE moving to curr_city
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 0] = last_city;
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 1] = 0;
                drops[i * MAX_DROPS * 3 + drop_idx * 3 + 2] = remaining_elder;
                drop_idx++;
                remaining_elder = 0;
                src = last_city; // evac rest from here next loop
                dropped_midway = true;
                cutoff_path_len = j; // up to (but not including) curr_city
                break;
            }

            total_dist += edge_len;
            last_city = curr_city;
        }


        // Append to global path array
        int append_len = dropped_midway ? cutoff_path_len : path_len;
        int start_idx = first_path ? 0 : 1; // skip repeating src
        for (int j = start_idx; j < append_len; ++j) {
            if (path_size[i] < MAX_PATH)
                paths[i * MAX_PATH + path_size[i]++] = temp_path[j];
        }
        first_path = false;

        if (dropped_midway) continue;

        // Try to drop at shelter
        int want = remaining_prime + remaining_elder;
        int* cap_ptr = &shelter_capacity[best_shelter];
        int give = 0, old_val, new_val;

        while (true) {
            old_val = atomicAdd(cap_ptr, 0);
            if (old_val == 0) break;

            give = min(want, old_val);
            new_val = old_val - give;

            int prev = atomicCAS(cap_ptr, old_val, new_val);
            if (prev == old_val) break;
        }

        if (give > 0) {
            int drop_elder = min(give, remaining_elder);
            give -= drop_elder;
            int drop_prime = min(give, remaining_prime);

            remaining_elder -= drop_elder;
            remaining_prime -= drop_prime;

            drops[i * MAX_DROPS * 3 + drop_idx * 3 + 0] = best_shelter;
            drops[i * MAX_DROPS * 3 + drop_idx * 3 + 1] = drop_prime;
            drops[i * MAX_DROPS * 3 + drop_idx * 3 + 2] = drop_elder;
            drop_idx++;

            src = best_shelter;
        } else {
            // Shelter full — try next
            continue;
        }
    }

    num_drops[i] = drop_idx;
    //path_size[i] = paths
}







int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]);
    ofstream outfile(argv[2]);

    int num_cities, num_roads;
    infile >> num_cities >> num_roads;

    // Graph structure
    int* edge_heads = new int[num_roads * 2];
    int* edge_tos = new int[num_roads * 2];
    int* edge_lens = new int[num_roads * 2];
    int* edge_next = new int[num_roads * 2];
    int* head = new int[num_cities];
    fill(head, head + num_cities, -1);

    int edge_count = 0;
    for (int i = 0; i < num_roads; i++) {
        int u, v, l, c;
        infile >> u >> v >> l >> c;

        edge_tos[edge_count] = v;
        edge_lens[edge_count] = l;
        edge_next[edge_count] = head[u];
        head[u] = edge_count++;
        
        edge_tos[edge_count] = u;
        edge_lens[edge_count] = l;
        edge_next[edge_count] = head[v];
        head[v] = edge_count++;
    }

    int num_shelters;
    infile >> num_shelters;

    int* is_shelter = new int[num_cities]();
    int* shelter_capacity = new int[num_cities]();

    for (int i = 0; i < num_shelters; i++) {
        int city, cap;
        infile >> city >> cap;
        is_shelter[city] = 1;
        shelter_capacity[city] = cap;
    }

    int num_populated;
    infile >> num_populated;

    int* pop_city = new int[num_populated];
    int* population_prime = new int[num_populated];
    int* population_elder = new int[num_populated];

    for (int i = 0; i < num_populated; i++) {
        infile >> pop_city[i] >> population_prime[i] >> population_elder[i];
        std::cout<<pop_city[i]<<population_prime[i]<<population_elder[i]<<std::endl;
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;

    // Device memory
    int *d_pop_city, *d_population_prime, *d_population_elder;
    int *d_edge_heads, *d_edge_tos, *d_edge_lens, *d_edge_next, *d_head;
    int *d_is_shelter, *d_shelter_capacity;

    cudaMalloc(&d_pop_city, num_populated * sizeof(int));
    cudaMalloc(&d_population_prime, num_populated * sizeof(int));
    cudaMalloc(&d_population_elder, num_populated * sizeof(int));
    cudaMalloc(&d_edge_heads, edge_count * sizeof(int));
    cudaMalloc(&d_edge_tos, edge_count * sizeof(int));
    cudaMalloc(&d_edge_lens, edge_count * sizeof(int));
    cudaMalloc(&d_edge_next, edge_count * sizeof(int));
    cudaMalloc(&d_head, num_cities * sizeof(int));
    cudaMalloc(&d_is_shelter, num_cities * sizeof(int));
    cudaMalloc(&d_shelter_capacity, num_cities * sizeof(int));

    cudaMemcpy(d_pop_city, pop_city, num_populated * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_population_prime, population_prime, num_populated * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_population_elder, population_elder, num_populated * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_heads, edge_heads, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_tos, edge_tos, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_lens, edge_lens, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_next, edge_next, edge_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_head, head, num_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_shelter, is_shelter, num_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelter_capacity, shelter_capacity, num_cities * sizeof(int), cudaMemcpyHostToDevice);

    // Output arrays
    long long* path_size = new long long[num_populated];
    long long* paths = new long long[num_populated * MAX_PATH];
    long long* num_drops = new long long[num_populated];
    long long* drops = new long long[num_populated * MAX_DROPS * 3];

    long long *d_path_size, *d_paths, *d_num_drops, *d_drops;
    cudaMalloc(&d_path_size, num_populated * sizeof(long long));
    cudaMalloc(&d_paths, num_populated * MAX_PATH * sizeof(long long));
    cudaMalloc(&d_num_drops, num_populated * sizeof(long long));
    cudaMalloc(&d_drops, num_populated * MAX_DROPS * 3 * sizeof(long long));

    // Kernel launch
    int blockSize = 256;
    int gridSize = (num_populated + blockSize - 1) / blockSize;
    evacuate_kernel<<<gridSize, blockSize>>>(num_populated, d_pop_city, d_population_prime, d_population_elder,
                                              d_edge_heads, d_edge_tos, d_edge_lens, d_edge_next, d_head,
                                              num_cities, d_is_shelter, d_shelter_capacity, max_distance_elderly,
                                              d_path_size, d_paths, d_num_drops, d_drops);

    cudaMemcpy(path_size, d_path_size, num_populated * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(paths, d_paths, num_populated * MAX_PATH * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_drops, d_num_drops, num_populated * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(drops, d_drops, num_populated * MAX_DROPS * 3 * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Synchronize and check for post-launch errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Post-Kernel Execution Error: " << cudaGetErrorString(err) << std::endl;
    }
    Output results
    for (int i = 0; i < num_populated; ++i) {
        for (int j = 0; j < path_size[i]; ++j)
            outfile << paths[i * MAX_PATH + j] << " ";
        outfile << "\n";
    }

    for (int i = 0; i < num_populated; ++i) {
        for (int j = 0; j < num_drops[i]; ++j) {
            outfile << drops[i * MAX_DROPS * 3 + j * 3 + 0] << " "
                    << drops[i * MAX_DROPS * 3 + j * 3 + 1] << " "
                    << drops[i * MAX_DROPS * 3 + j * 3 + 2] << " ";
        }
        outfile << "\n";

    // outfile << "path_sizes = [";
    // for (int i = 0; i < num_populated; i++) outfile << (i ? ", " : "") << path_size[i];
    // outfile << "]\npaths = [";
    // for (int i = 0; i < num_populated; i++) {
    //     outfile << (i ? ", [" : "[");
    //     for (int j = 0; j < path_size[i]; j++) outfile << (j ? ", " : "") << paths[i * MAX_PATH + j];
    //     outfile << "]";
    // }
    // outfile << "]\nnum_drops = [";
    // for (int i = 0; i < num_populated; i++) outfile << (i ? ", " : "") << num_drops[i];
    // outfile << "]\ndrops = [";
    // for (int i = 0; i < num_populated; i++) {
    //     outfile << (i ? ", [" : "[");
    //     for (int j = 0; j < num_drops[i]; j++) {
    //         int base = i * MAX_DROPS * 3 + j * 3;
    //         outfile << (j ? ", " : "") << "(" << drops[base] << ", " << drops[base + 1] << ", " << drops[base + 2] << ")";
    //     }
    //     outfile << "]";
    // }
    // outfile << "]\n";

    return 0;
}
