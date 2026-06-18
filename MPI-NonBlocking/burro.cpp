#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <map>

#define TAG_CARD 1
#define TAG_WINNER 2

int main(int argc, char** argv) {
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int right = (rank + 1) % n_procs;
    int left = (rank - 1 + n_procs) % n_procs;
    int hand[4];
    std::vector<int> full_deck;

    char filename[50];
    sprintf(filename, "game_log_rank_%d.txt", rank);
    FILE *log_file = fopen(filename, "w");

    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n_procs; i++) {
            for (int j = 0; j < 4; j++) full_deck.push_back(i);
        }
        std::random_shuffle(full_deck.begin(), full_deck.end());
    } else {
        full_deck.resize(n_procs * 4);
    }

    int winner = -1;
    double start_time = MPI_Wtime();
    MPI_Scatter(full_deck.data(), 4, MPI_INT, hand, 4, MPI_INT, 0, MPI_COMM_WORLD);

    fprintf(log_file, "Rank %d Initial Hand: [%d, %d, %d, %d]\n", rank, hand[0], hand[1], hand[2], hand[3]);
    fflush(log_file);

    while (winner == -1) {
        if (hand[0] == hand[1] && hand[1] == hand[2] && hand[2] == hand[3]) {
            winner = rank;
            for (int i = 0; i < n_procs; i++) {
                if (i != rank) MPI_Send(&winner, 1, MPI_INT, i, TAG_WINNER, MPI_COMM_WORLD);
            }
            break; 
        }

        std::map<int, int> counts;
        for(int i=0; i<4; i++) counts[hand[i]]++;
        int idx_to_send = 0;
        int min_c = 5;
        for(int i=0; i<4; i++) {
            if(counts[hand[i]] < min_c) { min_c = counts[hand[i]]; idx_to_send = i; }
        }

        int card_to_send = hand[idx_to_send];
        int card_received;
        int card_source = -1;
        MPI_Request s_req;
        MPI_Status status;

        MPI_Isend(&card_to_send, 1, MPI_INT, right, TAG_CARD, MPI_COMM_WORLD, &s_req);

        while (true) {
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (!flag) {
                continue;
            }

            if (status.MPI_TAG == TAG_WINNER) {
                MPI_Recv(&winner, 1, MPI_INT, status.MPI_SOURCE, TAG_WINNER, MPI_COMM_WORLD, &status);
                MPI_Wait(&s_req, MPI_STATUS_IGNORE);
                goto final_step;
            }

            if (status.MPI_TAG == TAG_CARD) {
                MPI_Recv(&card_received, 1, MPI_INT, status.MPI_SOURCE, TAG_CARD, MPI_COMM_WORLD, &status);
                card_source = status.MPI_SOURCE;
                break;
            }
        }

        MPI_Wait(&s_req, MPI_STATUS_IGNORE);
        
        fprintf(log_file, "Rank %d: Sent %d to Rank %d, Received %d from Rank %d\n", 
                rank, card_to_send, right, card_received, card_source);
        fflush(log_file);

        hand[idx_to_send] = card_received;
    }

final_step:
    MPI_Barrier(MPI_COMM_WORLD);

    // Resolve simultaneous local wins so the game reports only one winner.
    int winner_candidate = (winner == -1) ? n_procs : winner;
    int final_winner;
    MPI_Allreduce(&winner_candidate, &final_winner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    winner = final_winner;
    
    double end_time = MPI_Wtime();
    if (rank == winner) {
        printf("Rank %d WON in %f seconds!\n", rank, end_time - start_time);
    }

    if (log_file != NULL) {
        fclose(log_file);
    }

    MPI_Finalize();
    return 0;
}
