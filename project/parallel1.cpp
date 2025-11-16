#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>  
#include <chrono>
using namespace std::chrono;

using namespace std;

double upper(){ return 0; }
double bottom(double x){ return 0.5*x*(x+1.0); }
double left(){ return 0; }
double right(double y){return y*(2.0*y+1.0); }

int main(int argc, char* argv[]) {
    double dx = 1e-2, dy = 1e-2, eps = 1e-6, local_diff, global_diff;
    int N = 2 / dx, M = 1 / dy, ProcRank, ProcNum;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    auto start = high_resolution_clock::now();

    // Let MPI choose optimal topology
    int dims[2] = {0, 0};
    MPI_Dims_create(ProcNum, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    if (N % dims[0] != 0 || M % dims[1] != 0) {
        if (ProcRank == 0) {
            cerr << "Error: Grid size must be divisible by processor grid dimensions\n";
        }
        MPI_Finalize();
        return 1;
    }

    int local_N = N / dims[0];
    int local_M = M / dims[1];
    
    // Use 1D arrays for better performance
    vector<double> u((local_N + 2) * (local_M + 2), 0.0);
    vector<double> u_new((local_N + 2) * (local_M + 2), 0.0);

    auto idx = [&](int i, int j) { return i * (local_M + 2) + j; };

    int coords[2];
    MPI_Cart_coords(cart_comm, ProcRank, 2, coords);

    // Set boundary conditions
    if (coords[1] == 0) {
        for (int i = 0; i <= local_N+1; i++) {
            u[idx(i, 0)] = upper();
            u_new[idx(i, 0)] = upper();
        }
    }
    if (coords[1] == dims[1] - 1) {
        for (int i = 0; i <= local_N+1; i++) {
            double x = (i - 1 + coords[0] * local_N) * dx;
            u[idx(i, local_M + 1)] = bottom(x);
            u_new[idx(i, local_M + 1)] = bottom(x);
        }
    }
    if (coords[0] == 0) {
        for (int j = 0; j <= local_M+1; j++) {
            u[idx(0, j)] = left();
            u_new[idx(0, j)] = left();
        }
    }
    if (coords[0] == dims[0] - 1) {
        for (int j = 0; j <= local_M+1; j++) {
            double y = (j - 1 + coords[1] * local_M) * dy;
            u[idx(local_N + 1, j)] = right(y);
            u_new[idx(local_N + 1, j)] = right(y);
        }
    }

    int up_rank, down_rank, left_rank, right_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &up_rank, &down_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);

    // Buffers for communication
    vector<double> send_left(local_N), recv_left(local_N);
    vector<double> send_right(local_N), recv_right(local_N);
    vector<double> send_up(local_M), recv_up(local_M);
    vector<double> send_down(local_M), recv_down(local_M);

    MPI_Request requests[8];
    int request_count = 0;

    int iter = 0;
    do {
        local_diff = 0.0;
        request_count = 0;

        // Non-blocking communication for ghost cells
        if (up_rank != MPI_PROC_NULL) {
            for (int j = 1; j <= local_M; j++) {
                send_up[j-1] = u[idx(1, j)];
            }
            MPI_Isend(send_up.data(), local_M, MPI_DOUBLE, up_rank, 0, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_up.data(), local_M, MPI_DOUBLE, up_rank, 0, cart_comm, &requests[request_count++]);
        }

        if (down_rank != MPI_PROC_NULL) {
            for (int j = 1; j <= local_M; j++) {
                send_down[j-1] = u[idx(local_N, j)];
            }
            MPI_Isend(send_down.data(), local_M, MPI_DOUBLE, down_rank, 0, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_down.data(), local_M, MPI_DOUBLE, down_rank, 0, cart_comm, &requests[request_count++]);
        }

        if (left_rank != MPI_PROC_NULL) {
            for (int i = 1; i <= local_N; i++) {
                send_left[i-1] = u[idx(i, 1)];
            }
            MPI_Isend(send_left.data(), local_N, MPI_DOUBLE, left_rank, 0, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_left.data(), local_N, MPI_DOUBLE, left_rank, 0, cart_comm, &requests[request_count++]);
        }

        if (right_rank != MPI_PROC_NULL) {
            for (int i = 1; i <= local_N; i++) {
                send_right[i-1] = u[idx(i, local_M)];
            }
            MPI_Isend(send_right.data(), local_N, MPI_DOUBLE, right_rank, 0, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_right.data(), local_N, MPI_DOUBLE, right_rank, 0, cart_comm, &requests[request_count++]);
        }

        // Wait for all communication to complete
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        // Update ghost cells
        if (up_rank != MPI_PROC_NULL) {
            for (int j = 1; j <= local_M; j++) {
                u[idx(0, j)] = recv_up[j-1];
            }
        }
        if (down_rank != MPI_PROC_NULL) {
            for (int j = 1; j <= local_M; j++) {
                u[idx(local_N + 1, j)] = recv_down[j-1];
            }
        }
        if (left_rank != MPI_PROC_NULL) {
            for (int i = 1; i <= local_N; i++) {
                u[idx(i, 0)] = recv_left[i-1];
            }
        }
        if (right_rank != MPI_PROC_NULL) {
            for (int i = 1; i <= local_N; i++) {
                u[idx(i, local_M + 1)] = recv_right[i-1];
            }
        }

        // Jacobi iteration
        for (int i = 1; i <= local_N; i++) {
            for (int j = 1; j <= local_M; j++) {
                u_new[idx(i, j)] = 0.25 * (u[idx(i + 1, j)] + u[idx(i - 1, j)] + 
                                          u[idx(i, j + 1)] + u[idx(i, j - 1)]);
                local_diff = max(local_diff, fabs(u_new[idx(i, j)] - u[idx(i, j)]));
            }
        }

        // Swap arrays
        swap(u, u_new);

        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        iter++;
        
    } while (global_diff > eps && iter < 10000);

    // Gather results efficiently
    vector<double> local_data(local_N * local_M * 3);
    int data_idx = 0;
    for (int i = 1; i <= local_N; i++) {
        for (int j = 1; j <= local_M; j++) {
            double x = (i - 1 + coords[0] * local_N) * dx;
            double y = (j - 1 + coords[1] * local_M) * dy;
            local_data[data_idx++] = x;
            local_data[data_idx++] = y;
            local_data[data_idx++] = u[idx(i, j)];
        }
    }

    vector<double> global_data;
    vector<int> recv_counts(ProcNum), displs(ProcNum);
    int local_count = local_N * local_M * 3;

    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (ProcRank == 0) {
        int total = 0;
        for (int i = 0; i < ProcNum; i++) {
            displs[i] = total;
            total += recv_counts[i];
        }
        global_data.resize(total);
    }

    MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE, 
                global_data.data(), recv_counts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    if (ProcRank == 0) {
        cout << "Execution time: " << duration.count() << " microseconds" << endl;
        // cout << "Iterations: " << iter << endl;
        
        ofstream File("parallel.txt");
        for (size_t i = 0; i < global_data.size(); i += 3) {
            File << global_data[i] << "\t" << global_data[i+1] << "\t" << global_data[i+2] << "\n";
        }
        File.close();
    }

    MPI_Finalize();
    return 0;
}