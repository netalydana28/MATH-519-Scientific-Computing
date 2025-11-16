import pandas as pd
import numpy as np

def find_solution_error(sequential_file, parallel_file):

    try:

        df_seq = pd.read_csv(sequential_file, sep='\t', header=None, 
                             names=['x', 'y', 'u_seq'], usecols=[0, 1, 2])
        df_par = pd.read_csv(parallel_file, sep='\t', header=None, 
                             names=['y', 'x', 'u_par'], usecols=[0, 1, 2])
    
    except FileNotFoundError:
        print(f"Error: One or both files were not found.")
        return
    
    df_merged = pd.merge(df_seq, df_par, on=['x', 'y'], how='inner')
    # print(df_merged)
    if df_merged.empty:
        print("Error: The dataframes could not be merged. Check coordinate columns.")
        return

    df_merged['error'] = np.abs(df_merged['u_seq'] - df_merged['u_par'])

    max_error = df_merged['error'].max()
    mean_abs_error = df_merged['error'].mean()
    
    max_error_point = df_merged.loc[df_merged['error'].idxmax()]

    print("--- Error Analysis (Sequential vs Parallel) ---")
    print(f"Total points compared: {len(df_merged)}")
    print("-" * 40)
    print(f"Maximum Absolute Error: {max_error:.8f}")
    print(f"Mean Absolute Error: {mean_abs_error:.8f}")
    print("-" * 40)
    print("Coordinates of Maximum Error Point:")
    print(f"x: {max_error_point['x']:.4f}, y: {max_error_point['y']:.4f}")
    print(f"u_sequential: {max_error_point['u_seq']:.8f}")
    print(f"u_parallel:   {max_error_point['u_par']:.8f}")

find_solution_error('sequential.txt', 'parallel.txt')