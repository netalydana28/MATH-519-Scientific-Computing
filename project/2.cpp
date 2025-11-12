#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>  


using namespace std;
using namespace std::chrono;

int i, j, k;

double upper(){ return 0; }

double bottom(double x){ return 0.5*x*(x+1);}

double left(){ return 0;}

double right(double y){return y*(2.0*y+1.0);}

int main(){
    auto start = high_resolution_clock::now();
    double dx=1e-2, dy=1e-2, eps = 1e-6, max_diff;
    int N=2/dx, M=1/dy; 
    int N_time = 1e+2;


    double u[N][M], u_new[N][M];

    for(i=0; i<N; i++){
        for(j=0; j<M; j++){
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;

        }
    }


    for(i=0; i<N; i++){
        u[i][0] = upper();
        u[i][M-1] = bottom(i*dx);
        u_new[i][0] = upper();
        u_new[i][M-1] = bottom(i*dx);
    }

    for(j=0; j<M; j++){
        u[0][j] = left();
        u[N-1][j] = right(j*dy);
        u_new[0][j] = left();
        u_new[N-1][j] = right(j*dy);
    }
    do{
        max_diff = 0.0;
        for(i=1; i<N-1; i++){
            for(j=1; j<M-1; j++){
                u_new[i][j] = 0.25*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]);
                if(max_diff<fabs(u[i][j] - u_new[i][j])){
                    max_diff = fabs(u[i][j] - u_new[i][j]);
                }
                // u[i][j] = u_new[i][j];
            }
        }
        for(i=0; i<N; i++){
            for(j=0; j<M; j++){
                u[i][j] = u_new[i][j];

            }
        }
        cout<<max_diff<<"\n";
    }while(max_diff>eps);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
 
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    

    ofstream File("sequential.txt");
    // ofstream File1("analytical.txt");
    double x, y, u_num, u_an, max_error=0.0;    
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            x = j * dx;
            y = i * dy;
            u_num = u[i][j];
            // u_an = analytical(j*dx, i*dy, alpha, N_time*dt);
            // max_error = max(max_error, fabs(u_num - u_an));

            File<<x<<"\t"<<y<<"\t"<<u_num<< "\n";
            // File1<<x<<"\t"<<y<<"\t"<<u_an<< "\n";
        }
    }
    cout<<"Max_error "<<max_error<<'\n';
    File.close();
    // File1.close();

    return 0;
}