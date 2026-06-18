#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define N 10000

int main(int argc, char** argv) {
    int rank, n_procs;
    double *full_matrix = NULL;
    double *local_matrix = NULL;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    long int total_elements = (long int)N * N;
    long int n_elements_local = total_elements / n_procs;

    local_matrix = (double *)malloc(n_elements_local * sizeof(double));

    if (rank == 0) {
        full_matrix = (double *)malloc(total_elements * sizeof(double));
        srand(time(NULL));
        for (long int i = 0; i < total_elements; i++) {
            full_matrix[i] = (double)(rand() % 101);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    MPI_Scatter(full_matrix, n_elements_local, MPI_DOUBLE, 
                local_matrix, n_elements_local, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (long int i = 0; i < n_elements_local; i++) {
        local_sum += *(local_matrix + i);
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    double mean = global_sum / total_elements;

    if (mean != 0) {
        for (long int i = 0; i < n_elements_local; i++) {
            *(local_matrix + i) /= mean; 
        }
    }

    MPI_Gather(local_matrix, n_elements_local, MPI_DOUBLE, 
               full_matrix, n_elements_local, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Processes: %d | Time: %f seconds | Mean: %f\n", n_procs, end_time - start_time, mean);
        free(full_matrix);
    }

    free(local_matrix);
    MPI_Finalize();
    return 0;
}
