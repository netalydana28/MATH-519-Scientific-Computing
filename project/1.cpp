#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>  


using namespace std; 
int i, j, k;

double upper(){ return 0; }
double bottom(double x){ return 0.5*x*(x+1);}
double left(){ return 0;}
double right(double y){ return y*(2.0*y+1.0);}

void gather_strings(int rank, int size, string& local_str, string& global_str) {
    int local_len = local_str.length();
    vector<int> lengths(size, 0);
    MPI_Gather(&local_len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    int total_len = 0;
    if (rank == 0) {
        for (i = 0; i < size; ++i) {
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

    MPI_Gatherv(local_buffer.data(), local_len, MPI_CHAR, 
                global_buffer.data(), lengths.data(), displs.data(), MPI_CHAR, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_str.assign(global_buffer.begin(), global_buffer.end());
    }
}

int main(int argc, char* argv[]){
    double dx=1e-2, dy=1e-2, eps = 1e-6, local_diff, global_diff;
    int N=2/dx , M=1/dy, ProcRank, ProcNum; 
    double start, end;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if(ProcRank == 0){
        start = MPI_Wtime();
    }
    int dims[2] = {2, 2}, periods[2] = {0, 0};        
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // MPI_Request request_send, request_recv;

    int local_N = N / dims[0];
    int local_M = M / dims[1];
    vector<vector<double>> u(local_N + 2, vector<double>(local_M + 2, 0.0));
    vector<vector<double>> u_new(local_N + 2, vector<double>(local_M + 2, 0.0));

    for(i=0; i<local_N; i++){
        for(j=0; j<local_M; j++){
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }

    int coords[2];
    MPI_Cart_coords(cart_comm, ProcRank, 2, coords);

    if (coords[1] == 0) {
        for (i = 1; i <= local_N; ++i) {;
            u[i][0] = upper();
            u_new[i][0] = upper();
        }
    }
    if (coords[1] == dims[1] - 1) {
        for (i = 1; i <= local_N; ++i) {
            u[i][local_M] = bottom((i - 1 + coords[0] * local_N) * dx);
            u_new[i][local_M] = bottom((i - 1 + coords[0] * local_N) * dx);
        }
    }

    if (coords[0] == 0) {
        for (j = 1; j <= local_M; ++j) {
            u[0][j] = left();
            u_new[0][j] = left();
        }
    }
    if (coords[0] == dims[0] - 1) {
        for (j = 1; j <= local_M; ++j) {
            u[local_N][j] = right((j - 1 + coords[1] * local_M) * dy);
            u_new[local_N][j] = right((j - 1 + coords[1] * local_M) * dy);
        }
    }

    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
    // vector<double> left_column(local_N), right_column(local_N), left_ghost(local_N), right_ghost(local_N);

    do{
        // MPI_Request requests[8];
        if(ProcRank/sqrt(ProcNum) != sqrt(ProcNum) -1){
            MPI_Send(&u[local_N][*], local_M+2, MPI_DOUBLE, down, 0, cart_comm);
            MPI_Recv(&u[local_N+1][*], local_M+2, MPI_DOUBLE, down, 0, cart_comm, MPI_STATUS_IGNORE);
        }
        if(ProcRank/sqrt(ProcNum) != 0){
            MPI_Recv(&u[0][*], local_M+2, MPI_DOUBLE, up, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Send(&u[1][*], local_M+2, MPI_DOUBLE, up, 0, cart_comm);
        }

        if(ProcRank%sqrt(ProcNum) != sqrt(ProcNum) -1){
            MPI_Send(&u[*][local_M], local_N+2, MPI_DOUBLE, right, 0, cart_comm);
            MPI_Recv(&u[*][local_M+1], local_M+2, MPI_DOUBLE, right, 0, cart_comm, MPI_STATUS_IGNORE);
        }
        if(ProcRank%sqrt(ProcNum) != 0){
            MPI_Recv(&u[*][0], local_N+2, MPI_DOUBLE, left, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Send(&u[*][1], local_N+2, MPI_DOUBLE, left, 0, cart_comm);
        }

        local_diff = 0.0;
        for (i = 1; i <= local_N; ++i) {
            for (j = 1; j <= local_M; ++j) {
                u_new[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]);
                local_diff = max(local_diff, fabs(u_new[i][j] - u[i][j]));
            }
        }

        for (i = 1; i <= local_N; ++i) {
            for (j = 1; j <= local_M; ++j) {
                u[i][j] = u_new[i][j];
            }
        }
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    }while(global_diff > eps);

    // ofstream file("result_" + to_string(rank) + ".txt");
    string local_result="";
    double x, y;
    for (i = 1; i <= local_N; ++i) {
        for (j = 1; j <= local_M; ++j) {
            x = (i - 1 + coords[0] * local_N) * dx;
            y = (j - 1 + coords[1] * local_M) * dy;
            local_result  += to_string(x) + "\t" + to_string(y) + "\t" + to_string(u[i][j]) + "\n";
        }
    }
    string global_result;

    gather_strings(ProcRank, ProcNum, local_result, global_result);
    // file.close();
    
    // MPI_Gatherv(u_partition, local_ * N, MPI_DOUBLE, u, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
        // double maxError = 0.0;
        // end = MPI_Wtime();
        ofstream File("numerical.txt");
    // }
        // ofstream File1("analytical.txt");
        // double x, y, u_num;    
        // for (i = 1; i < local_N-1; i++) {
        //     for (j = 1; j < local_M-1; j++) {
        //         x = (i * dx)+(local_N*ProcRank);
        //         y = (j * dy)+(local_M*ProcRank);
        //         u_num = u[i][j];
        //         // u_an = analytical(j*dx, i*dy, alpha, N_time*dt);
        //         // max_error = max(max_error, fabs(u_num - u_an));

        //         File<<x<<"\t"<<y<<"\t"<<u_num<< "\n";
        //         // File1<<x<<"\t"<<y<<"\t"<<u_an<< "\n";
        //     }
        // }
        // cout<<"Max_error "<<max_error<<'\n';
    // if (ProcRank == 0) {
        // cout<<"Time taken by function: "<<(end - start)*1e+6<<"microseconds\n";
        File << global_result;
        File.close();
        // File1.close();
    }
    MPI_Finalize();
    return 0;
}