#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 10000
#define IDX(row, col) ((row) * N + (col))
#define TAG_COORDS 0
#define TAG_CROSS 1
#define TAG_SUM 2

int main(int argc, char *argv[])
{
    int rank, size;
    int i = 0, j = 0, k = 0, t = 0;
    double *matrix = NULL;
    double *cross_buffer = NULL;
    double start_time = 0.0, end_time;

    MPI_Datatype cross_type = MPI_DATATYPE_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: program requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        if (size != 5 && size != 10 && size != 15) {
            fprintf(stderr,
                    "Warning: assignment requires runs with 5, 10, and 15 processes.\n"
                    "You are currently running with %d processes.\n", size);
        }
    }

    if (rank == 0) {
        matrix = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
        if (matrix == NULL) {
            fprintf(stderr, "Error: cannot allocate matrix %dx%d.\n", N, N);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand((unsigned int)time(NULL));

        for (long long idx = 0; idx < (long long)N * (long long)N; idx++) {
            matrix[idx] = (double)(rand() % 11);
        }

        i = rand() % (N - 1);
        j = i + 1 + rand() % (N - 1 - i);

        k = rand() % (N - 1);
        t = k + 1 + rand() % (N - 1 - k);

        printf("Master: N=%d, i=%d, j=%d, k=%d, t=%d\n", N, i, j, k, t);
        fflush(stdout);

        start_time = MPI_Wtime();
    }

    int int_pack_size;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
    int coords_pack_size = 4 * int_pack_size;

    if (rank == 0) {
        void *coords_pack = malloc((size_t)coords_pack_size);
        if (coords_pack == NULL) {
            fprintf(stderr, "Master: cannot allocate coordinates pack buffer.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int position = 0;
        MPI_Pack(&i, 1, MPI_INT, coords_pack, coords_pack_size, &position, MPI_COMM_WORLD);
        MPI_Pack(&j, 1, MPI_INT, coords_pack, coords_pack_size, &position, MPI_COMM_WORLD);
        MPI_Pack(&k, 1, MPI_INT, coords_pack, coords_pack_size, &position, MPI_COMM_WORLD);
        MPI_Pack(&t, 1, MPI_INT, coords_pack, coords_pack_size, &position, MPI_COMM_WORLD);

        for (int r = 1; r < size; r++) {
            MPI_Send(coords_pack, position, MPI_PACKED, r, TAG_COORDS, MPI_COMM_WORLD);
        }

        free(coords_pack);
    } else {
        void *coords_pack = malloc((size_t)coords_pack_size);
        if (coords_pack == NULL) {
            fprintf(stderr, "Rank %d: cannot allocate coordinates pack buffer.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Status status;
        MPI_Recv(coords_pack, coords_pack_size, MPI_PACKED, 0, TAG_COORDS, MPI_COMM_WORLD, &status);

        int count;
        MPI_Get_count(&status, MPI_PACKED, &count);

        int position = 0;
        MPI_Unpack(coords_pack, count, &position, &i, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(coords_pack, count, &position, &j, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(coords_pack, count, &position, &k, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(coords_pack, count, &position, &t, 1, MPI_INT, MPI_COMM_WORLD);

        free(coords_pack);
    }

    int v_len = j - i + 1;
    int h_len = t - k + 1;
    int total_elements = v_len + h_len - 1;

    cross_buffer = (double *)malloc((size_t)total_elements * sizeof(double));
    if (cross_buffer == NULL) {
        fprintf(stderr, "Rank %d: cannot allocate cross_buffer of size %d.\n",
                rank, total_elements);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        int *blocklengths = (int *)malloc((size_t)v_len * sizeof(int));
        int *displacements = (int *)malloc((size_t)v_len * sizeof(int));

        if (blocklengths == NULL || displacements == NULL) {
            fprintf(stderr, "Master: cannot allocate indexed type arrays.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int n_blocks = 0;
        blocklengths[n_blocks] = h_len;
        displacements[n_blocks] = IDX(i, k);
        n_blocks++;

        for (int row = i + 1; row <= j; row++) {
            blocklengths[n_blocks] = 1;
            displacements[n_blocks] = IDX(row, k);
            n_blocks++;
        }

        MPI_Type_indexed(n_blocks,
                         blocklengths,
                         displacements,
                         MPI_DOUBLE,
                         &cross_type);
        MPI_Type_commit(&cross_type);

        free(blocklengths);
        free(displacements);

        for (int r = 1; r < size; r++) {
            MPI_Send(matrix, 1, cross_type, r, TAG_CROSS, MPI_COMM_WORLD);
        }

        int pos = 0;
        for (int col = k; col <= t; col++) {
            cross_buffer[pos++] = matrix[IDX(i, col)];
        }
        for (int row = i + 1; row <= j; row++) {
            cross_buffer[pos++] = matrix[IDX(row, k)];
        }
    } else {
        MPI_Recv(cross_buffer, total_elements, MPI_DOUBLE, 0, TAG_CROSS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double local_sum = 0.0;
    for (int idx = 0; idx < total_elements; idx++) {
        local_sum += cross_buffer[idx];
    }

    if (rank == 0) {
        printf("Master: local sum = %f\n", local_sum);

        for (int r = 1; r < size; r++) {
            double remote_sum;
            MPI_Recv(&remote_sum, 1, MPI_DOUBLE, r, TAG_SUM,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Master: received sum from rank %d = %f\n", r, remote_sum);
        }

        end_time = MPI_Wtime();
        printf("Elapsed time (steps 3-7): %f seconds\n", end_time - start_time);
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, TAG_SUM, MPI_COMM_WORLD);
    }

    free(cross_buffer);
    if (rank == 0) {
        free(matrix);
    }

    if (rank == 0) {
        MPI_Type_free(&cross_type);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
