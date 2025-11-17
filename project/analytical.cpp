#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

const double Lx = 2.0;  // x from 0 to 2
const double Ly = 1.0;  // y from 0 to 1

// Boundary conditions
double bottom_bc(double x) { return 0.5*x*(x+1); }  // u(x,1) = 0.5*x*(x+1)
double right_bc(double y)  { return y*(2.0*y+1.0); } // u(2,y) = y*(2*y+1)
// Other boundaries: u(x,0) = 0, u(0,y) = 0

// ============================================================
// ANALYTICAL SOLUTION FOR LAPLACE'S EQUATION
// ============================================================

// Fourier coefficient for bottom boundary condition
double An_coefficient(int n) {
    int max_points = 201;
    double sum = 0.0;
    double dx = Lx / (max_points-1);
    
    for (int i = 0; i < max_points; i++) {
        double x = i * dx;
        double f_x = bottom_bc(x);
        // if (x == 2.0){
        //     cout<<f_x<<"fkskkd\n";
        // }
        sum += f_x * sin(n * M_PI * x / Lx) * dx;
    }
    
    return (2.0 / Lx) * sum / sinh(n * M_PI * Ly / Lx);
}

// Fourier coefficient for right boundary condition  
double Bn_coefficient(int n) {
    int max_points = 101;
    double sum = 0.0;
    double dy = Ly / (max_points-1);
    
    for (int i = 0; i < max_points; i++) {
        double y = i * dy;
        double f_y = right_bc(y);
        // if (y == 1.0){
        //     cout<<f_y<<"fmkkskkd\n";
        // }
        sum += f_y * sin(n * M_PI * y / Ly) * dy;
    }
    
    return (2.0 / Ly) * sum / sinh(n * M_PI * Lx / Ly);
}

// Full analytical solution using separation of variables
double analytical_solution(double x, double y, int N_terms = 100) {
    if (x == Lx && y == Ly) return 3.0;  // Corner (2,1)
    if (x == Lx && y == 0) return 0.0;   // Corner (2,0)
    if (x == 0 && y == Ly) return 0.0;
    double u = 0.0;
    // Solution from right boundary u(2,y) = y*(2*y+1)
    for (int n = 1; n <= N_terms; n++) {
        double Bn = Bn_coefficient(n);
        u += Bn * sin(n * M_PI * y / Ly) * sinh(n * M_PI * x / Ly);
    }
    // Solution from bottom boundary u(x,1) = 0.5*x*(x+1)
    for (int n = 1; n <= N_terms; n++) {
        double An = An_coefficient(n);
        u += An * sin(n * M_PI * x / Lx) * sinh(n * M_PI * y / Lx);
    }
    
    
    
    return u;
}

// ============================================================
// VERIFICATION FUNCTIONS
// ============================================================

void verify_boundary_conditions(int N_terms) {
    cout << "Verifying boundary conditions with " << N_terms << " terms:\n";
    
    // Check bottom boundary (y = 1)
    cout << "\nBottom boundary (y=1):\n";
    for (double x = 0; x <= Lx; x += 0.5) {
        double analytical = analytical_solution(x, Ly, N_terms);
        double expected = bottom_bc(x);
        cout << "u(" << x << ",1) = " << analytical 
             << " (expected: " << expected 
             << ", error: " << fabs(analytical - expected) << ")\n";
    }
    
    // Check right boundary (x = 2)
    cout << "\nRight boundary (x=2):\n";
    for (double y = 0; y <= Ly; y += 0.25) {
        double analytical = analytical_solution(Lx, y, N_terms);
        double expected = right_bc(y);
        cout << "u(2," << y << ") = " << analytical 
             << " (expected: " << expected 
             << ", error: " << fabs(analytical - expected) << ")\n";
    }
    
    // Check zero boundaries
    cout << "\nZero boundaries:\n";
    cout << "u(0,0.5) = " << analytical_solution(0, 0.5, N_terms) << " (should be ~0)\n";
    cout << "u(1,0) = " << analytical_solution(1, 0, N_terms) << " (should be ~0)\n";
}

// ============================================================
// MAIN PROGRAM
// ============================================================


int main() {
    double dx = 0.01, dy = 0.01;
    int Nx = static_cast<int>(Lx / dx) + 1;
    int Ny = static_cast<int>(Ly / dy) + 1;
    
    // Verify the solution first
    verify_boundary_conditions(100);
    
    // Generate analytical solution data
    ofstream file("analytical.txt");
    
    cout << "\nGenerating analytical solution on " << Nx << " x " << Ny << " grid..." << endl;
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = i * dx;
            double y = j * dy;
            double u_an = analytical_solution(x, y, 100);
            
            file << x << "\t" << y << "\t" << u_an << "\n";
        }
    }
    
    file.close();
    cout << "Analytical solution saved to 'analytical_solution_fixed.txt'" << endl;
    
    // Test convergence with number of terms
    cout << "\nTesting convergence at (1.0, 0.5):\n";
    vector<int> terms = {10, 20, 50, 100, 200};
    for (int n : terms) {
        double u_val = analytical_solution(1.0, 0.5, n);
        cout << "N_terms = " << n << ": u = " << u_val << endl;
    }
    
    return 0;
}
