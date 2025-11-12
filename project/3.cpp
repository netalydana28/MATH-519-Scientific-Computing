#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>  

using namespace std;

int i, j, k;

double upper(){ return 0; }

double bottom(double x){ return 0.5*x*(x+1.0);}

double left(){ return 0;}

double right(double y){return y*(2.0*y+1.0);}

void gather_strings(int rank, int size, string& local_str, string& global_str) {
    int local_len = local_str.length();
    vector<int> lengths(size, 0);
    MPI_Gather(&local_len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    int total_len = 0;
    if (rank == 0) {
        for (i = 0; i < size; i++) {
            displs[i] = total_len;
            total_len += lengths[i];
        }
    }

    vector<char> local_buffer(local_len);
    copy(local_str.begin(), local_str.end(), local_buffer.begin());

    vector<char> global_buffer;
    if (rank == 0) {
        global_buffer.resize(total_len);
    }

    MPI_Gatherv(local_buffer.data(), local_len, MPI_CHAR, global_buffer.data(), lengths.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_str.assign(global_buffer.begin(), global_buffer.end());
    }
}

int main(int argc, char* argv[]) {
    double dx = 1e-2, dy = 1e-2, eps = 1e-6, local_diff, global_diff;
    int N = 2 / dx, M = 1 / dy, ProcRank, ProcNum;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        start = MPI_Wtime();
    }

    int dims[2] = {2, 2}, periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    if (N % dims[0] != 0 || M % dims[1] != 0) {
        if (ProcRank == 0) {
            cerr << "Error: Grid size must be divisible by dims[0] and dims[1]\n";
        }
        MPI_Finalize();
        return 0;
    }

    int local_N = N / dims[0];
    int local_M = M / dims[1];
    vector<vector<double>> u(local_N + 2, vector<double>(local_M + 2, 0.0));
    vector<vector<double>> u_new(local_N + 2, vector<double>(local_M + 2, 0.0));

    int coords[2];
    for(i=0; i<local_N; i++){
        for(j=0; j<local_M; j++){
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }
    MPI_Cart_coords(cart_comm, ProcRank, 2, coords);

    if (coords[1] == 0) {
        for (i = 0; i <= local_N+1; i++) {
            u[i][0] = upper();
            u_new[i][0] = upper();
        }
    }
    if (coords[1] == dims[1] - 1) {
        for (i = 0; i <= local_N+1; i++) {
            u[i][local_M + 1] = bottom((i - 1 + coords[0] * local_N) * dx);
            u_new[i][local_M + 1] = bottom((i - 1 + coords[0] * local_N) * dx);
            // cout<<u[i][local_M + 1];
        }
    }
    if (coords[0] == 0) {
        for (j = 0; j <= local_M+1; j++) {
            u[0][j] = left();
            u_new[0][j] = left();
        }
    }
    if (coords[0] == dims[0] - 1) {
        for (j = 0; j <= local_M+1; j++) {
            u[local_N + 1][j] = right((j - 1 + coords[1] * local_M) * dy);
            u_new[local_N + 1][j] = right((j - 1 + coords[1] * local_M) * dy);
        }
    }

    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    vector<double> left_col(local_N + 2), right_col(local_N + 2), left_ghost(local_N + 2), right_ghost(local_N + 2);
    do {
        if (down != MPI_PROC_NULL) {
            MPI_Send(u[local_N].data(), local_M + 2, MPI_DOUBLE, down, 0, cart_comm);
            MPI_Recv(u[local_N + 1].data(), local_M + 2, MPI_DOUBLE, down, 0, cart_comm, MPI_STATUS_IGNORE);
        }
        if (up != MPI_PROC_NULL) {
            MPI_Recv(u[0].data(), local_M + 2, MPI_DOUBLE, up, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Send(u[1].data(), local_M + 2, MPI_DOUBLE, up, 0, cart_comm);
        }

        for (i = 0; i < local_N + 2; i++) {
            left_col[i] = u[i][1];
            right_col[i] = u[i][local_M];
        }

        if (right != MPI_PROC_NULL) {
            MPI_Send(right_col.data(), local_N + 2, MPI_DOUBLE, right, 0, cart_comm);
            MPI_Recv(right_ghost.data(), local_N + 2, MPI_DOUBLE, right, 0, cart_comm, MPI_STATUS_IGNORE);
            for (i = 0; i < local_N + 2; i++) {
                u[i][local_M+1] = right_ghost[i];
            }
        }
        if (left != MPI_PROC_NULL) {
            MPI_Recv(left_ghost.data(), local_N + 2, MPI_DOUBLE, left, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Send(left_col.data(), local_N + 2, MPI_DOUBLE, left, 0, cart_comm);
            for (i = 0; i < local_N + 2; i++) {
                u[i][0] = left_ghost[i];
            }
        }
        // for (i = 0; i < local_N + 2; i++) {
        //     u[i][0] = left_ghost[i];
        //     u[i][local_M+1] = right_ghost[i];
        // }

        local_diff = 0.0;
        for (i = 1; i <= local_N; i++) {
            for (j = 1; j <= local_M; j++) {
                u_new[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]);
                local_diff = max(local_diff, fabs(u_new[i][j] - u[i][j]));
            }
        }
        for (i = 1; i <= local_N; i++) {
            for (j = 1; j <= local_M; j++) {
                u[i][j] = u_new[i][j];
            }
        }
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        cout<<global_diff<<"\n";
    } while (global_diff > eps);

    if (ProcRank == 0) {
        end = MPI_Wtime();
        // printf("Time taken by function: %.5f microseconds\n", (end - start)*1e+6);
        cout<<"Time taken by function: "<<(end - start)*1000000<<" microseconds\n";
    }

    string local_result = "";
    double x, y;
    for (i = 1; i <= local_N; i++) {
        for (j = 1; j <= local_M; j++) {
            x = (i - 1 + coords[0] * local_N) * dx;
            y = (j - 1 + coords[1] * local_M) * dy;
            local_result += to_string(x) + "\t" + to_string(y) + "\t" + to_string(u[i][j]) + "\n";
        }
    }

    string global_result;
    gather_strings(ProcRank, ProcNum, local_result, global_result);

    if (ProcRank == 0) {
        // end = MPI_Wtime();
        // printf("Time taken by function: %.5f microseconds\n", (end - start)*1e+6);
        // cout<<"Time taken by function: "<<(end - start)*1e+6<<" microseconds\n";
        ofstream File("parallel.txt");
        File << global_result;
        File.close();
        
    }

    MPI_Finalize();
    return 0;
}
