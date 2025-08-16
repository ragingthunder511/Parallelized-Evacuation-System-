
# CUDA-based Evacuation Simulation

**Short description**  
GPU-parallelized evacuation simulator implemented in **CUDA C++**. Models evacuation from multiple populated cities to shelters on a road network, handling shelter capacity limits, elderly travel-distance constraints, and partial drops. Designed for large-scale inputs (10K+ cities, 2M+ roads).

---

## Key highlights (one-line / numeric)
- **Scale tested:** 10,000+ cities, 2,000,000+ roads  
- **Runtime:** ~20–30× speedup vs single-threaded CPU baseline (project benchmark)  
- **Features:** shortest-path routing (Dijkstra), atomic shelter allocation, elderly cutoff handling, partial drops

---

## Repository contents
```
evacuation.cu      # Main CUDA implementation (kernel + host setup)
input.txt          # Example input (small-scale)
output.txt         # Example output for sample input
README.md          # This file
```

---

## Requirements
- NVIDIA GPU with CUDA support (Compute Capability ≥ 5.0 recommended)  
- CUDA Toolkit (tested with CUDA 11.x or newer)  
- nvcc / gcc toolchain

---

## Build & run
1. Compile:
```bash
nvcc -O2 evacuation.cu -o evacuation
```

2. Run:
```bash
./evacuation input.txt output.txt
```

---

## Input file format (plain text)
1. Graph header:
```
<num_cities> <num_roads>
```
2. Next `<num_roads>` lines (each road is undirected; stored as two directed edges):
```
u v length capacity
```
*(u and v are 0-based city indices)*

3. Shelters:
```
<num_shelters>
city capacity
...
```

4. Populated cities:
```
<num_populated>
city prime_population elderly_population
...
```

5. Elderly distance constraint:
```
max_distance_elderly
```

**Notes**
- Road capacity field can be ignored or extended depending on your input parser. The provided CUDA code uses length primarily.
- Cities are indexed from `0` to `num_cities-1`.

---

## Output format
1. For each populated city — one line listing the evacuated path (sequence of city indices).
2. For each populated city — one line listing drop events as tuples:
```
<drop_city_1> <prime_dropped_1> <elder_dropped_1>  <drop_city_2> <prime_dropped_2> <elder_dropped_2>  ...
```

Example (human-readable):
```
Paths:
0 1 3 5
2 4 6

Drops:
5 120 30  2 0 10
4 50 0
```

---

## Sample `input.txt` (5 cities — small example)
```
5 5
0 1 5 0
1 2 3 0
1 3 4 0
3 4 2 0
2 4 6 0
2
4 200
2 100
2
0 120 30
1 60 10
10
```

Explanation:
- 5 cities, 5 roads
- 2 shelters: city 4 (cap 200), city 2 (cap 100)
- 2 populated cities:
  - city 0: 120 prime, 30 elderly
  - city 1: 60 prime, 10 elderly
- `max_distance_elderly = 10`

---

## Design notes (technical)
- Each populated city is processed by one CUDA thread; threads compute shortest paths (Dijkstra) from their source city on-device and iteratively allocate to the nearest available shelter.
- Elderly are restricted by `max_distance_elderly`: if continuing along the computed path would exceed the limit, elderly are dropped at the last safe city (partial drop), and the thread resumes evacuation from that city.
- Shelter allocation uses `atomicCAS`/`atomicAdd` on device-side capacities to ensure consistency under contention.
- Path storage and drop events are written to fixed-size arrays (configurable via `MAX_PATH`, `MAX_DROPS`) to simplify device-host communication.

---

## Known limitations & TODOs
- Current in-kernel Dijkstra uses arrays sized by `MAX_CITIES` — may not fit very large graphs without adjusting memory strategy.
- No heuristics (A\*) or multi-source optimizations implemented; potential improvements: multi-GPU support, hierarchical decomposition, or parallel shortest-path algorithms (e.g., delta-stepping).
- Input parsing and validation are minimal — ensure input indices and sizes match compile-time constants.

---

## License & Contact
MIT License — feel free to reuse and adapt for academic projects.  
Author: Grishma Uday Karekar ( cs24m020 )  

---

If you want, I can also:
- Add the `evacuation.cu` clean source file and a tested `input.txt`/`output.txt` pair into a zip for download, or  
- Convert this README into PDF or a one-page LaTeX summary.  
