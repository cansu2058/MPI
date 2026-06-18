# MPI Derived Datatypes Assignment

This project implements a parallel MPI program that distributes a cross-shaped region of a 10000x10000 matrix from a master process to all slaves using MPI derived datatypes. The random indices are sent with packed communication, while the cross data is sent directly with an indexed derived datatype.

## Project Overview

The master process generates a random NxN matrix and selects a cross-shaped subregion defined by 4 random indices. It packs and sends those indices to the slave processes, creates a custom indexed MPI derived datatype representing the cross, then sends the cross data to all slaves, which each compute the sum of the received elements.

## Technical Requirements Met

- **`MPI_Pack` / `MPI_Unpack`**: Used to pack and send the random indices `i`, `j`, `k`, and `t` to the slaves in a single transmission.
- **`MPI_Type_indexed`**: Used to define the complete cross-shaped region directly with block lengths and displacements.
- **Derived datatype transmission**: The master sends the matrix cross directly with the indexed datatype; the cross itself is not packed.
- **Point-to-point MPI communication**: Slaves receive the cross data, compute the sum, and send the result back to the master.

## Program Logic

1. Master allocates and fills a 10000x10000 matrix with random values between 0 and 10.
2. Master generates 4 random indices i, j, k, t (where i<j and k<t), packs them, and sends them to all slaves.
3. Master creates an indexed derived datatype representing the cross shape in the matrix.
4. Master sends the cross data directly to all slaves using the indexed datatype.
5. Each slave computes the sum of all received elements.
6. All slaves send their sum back to the master, which prints all results and elapsed time.

## Compile & Run

```bash
source /soft/icelake/intel/oneapi/setvars.sh
export UCX_NET_DEVICES=mlx5_0:1
cp Datos.cpp Datos.c
mpicc Datos.c -o output
mpirun -np 5 ./output   # also tested with 10 and 15 processes
```

## Performance Analysis (SCAYLE Cluster)

The following results were obtained using the **Caléndula (SCAYLE)** Supercomputing Cluster with N=10000, taking the minimum of 5 runs per process count:

| Process Count (NP) | Min Elapsed Time (s) |
|:------------------:|:--------------------:|
| 5                  | 0.000679             |
| 10                 | 0.001316             |
| 15                 | 0.002743             |

*Note: Execution time increases with process count due to the added communication overhead of sending the indexed cross data to more slave processes.*

## Repository Contents

- `Datos.cpp`: The complete MPI source code utilizing packed indices and an indexed derived datatype.
- `5.png` `10.png` `15.png`: Screenshots of the program running on the Caléndula cluster.
- `graph.png`: Performance graph comparing elapsed times across process counts.
- `execution_times.txt`: Full output from the 5 runs for each process count.
