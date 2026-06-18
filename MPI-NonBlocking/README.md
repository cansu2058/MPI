# MPI Burro Game - Non-Blocking Implementation

This project implements a parallel version of the "Burro" (Donkey) card game using the **Message Passing Interface (MPI)**. The core objective is to demonstrate high-performance distributed computing through non-blocking communication patterns.

## Project Overview
In this implementation, players (MPI processes) are arranged in a ring topology. Each player starts with 4 random cards and attempts to collect 4 of a kind by passing unwanted cards to their neighbor.

## Technical Requirements Met
The implementation strictly adheres to non-blocking MPI standards:
- **`MPI_Iprobe`**: Used inside a loop with `MPI_ANY_SOURCE` and `MPI_ANY_TAG` to wait until a message is available.
- **Message tag handling**: Once `MPI_Iprobe` detects a message, `status.MPI_TAG` is checked to determine whether the message is a card (`TAG_CARD`) or a winner notification (`TAG_WINNER`).
- **`MPI_Isend` & `MPI_Wait`**: Card transfers are initiated with non-blocking sends and completed safely before continuing.
- **Winner agreement**: A final `MPI_Allreduce` is used to resolve simultaneous local wins so that only one final winner is reported.

## Game Logic
1. **Shuffling & Scattering**: Rank 0 creates the deck and distributes 4 cards to each rank using `MPI_Scatter`.
2. **The Exchange**: Each rank identifies its least frequent card and sends it to the `(rank + 1) % size` neighbor while receiving from `(rank - 1 + size) % size`.
3. **Winning Condition**: The first rank to collect 4 identical cards sends a broadcast signal to all other ranks.
4. **Termination**: Processes use `MPI_Iprobe` with `MPI_ANY_SOURCE` and `MPI_ANY_TAG` during their exchange loop to detect either a card or a winner signal and exit gracefully.

## Performance Analysis (SCAYLE Cluster)
The following results were obtained using the **Calendula (SCAYLE)** Supercomputing Cluster:

| Process Count (NP) | Winner Rank | Execution Time (Seconds) |
|:------------------:|:-----------:|:------------------------:|
| 8                  | Rank 0      | 0.323931 s               |
| 16                 | Rank 7      | 0.740987 s               |
| 32                 | Rank 6      | 2.271239 s               |

*Note: Execution time increases with process count due to the added communication overhead and synchronization required for a larger ring topology.*

## Repository Contents
- `burro.cpp`: The complete C++ source code utilizing MPI.
- `initial_hands_np8.txt`, `initial_hands_np16.txt`, `initial_hands_np32.txt`: Initial hands for each process count.
- `all_moves_np8.txt`, `all_moves_np16.txt`, `all_moves_np32.txt`: Comprehensive logs containing the initial hands and every move made during each execution.
- `execution_times.txt`: A summary of performance metrics for 8, 16, and 32 processes.
