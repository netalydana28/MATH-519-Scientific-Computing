import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def visualize_solution_error(sequential_file, parallel_file):

    try:
        df_seq = pd.read_csv(sequential_file, sep='\t', header=None, 
                             names=['x', 'y', 'u_seq'], usecols=[0, 1, 2])
        df_par = pd.read_csv(parallel_file, sep='\t', header=None, 
                             names=['x', 'y', 'u_par'], usecols=[0, 1, 2])
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    df_merged = pd.merge(df_seq, df_par, on=['x', 'y'], how='inner')

    if df_merged.empty:
        print("Error: Dataframes could not be merged. Check coordinate columns in files.")
        return

    df_merged['error'] = np.abs(df_merged['u_seq'] - df_merged['u_par'])
    
    x = df_merged['x'].values
    y = df_merged['y'].values
    u_seq = df_merged['u_seq'].values
    u_par = df_merged['u_par'].values
    error = df_merged['error'].values

    max_error = np.max(error)
    print(f"Max Absolute Error between solutions: {max_error:.8e}")


    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Sequential Solution
    ax = axes[0]
    scatter1 = ax.scatter(x, y, c=u_seq, cmap="viridis", s=15)
    fig.colorbar(scatter1, ax=ax, label='Potential U')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("1. Sequential Solution")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 2. Parallel Solution
    ax = axes[1]
    scatter2 = ax.scatter(x, y, c=u_par, cmap="viridis", s=15)
    fig.colorbar(scatter2, ax=ax, label='Potential U')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2. Analytical Solution")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 3. Absolute Error
    ax = axes[2]

    if max_error > 1e-6:
        norm = None
        cbar_label = 'Absolute Error |U_seq - U_an|'
    else:
        norm = LogNorm(vmin=error[error > 0].min(), vmax=max_error)
        cbar_label = 'Absolute Error (Log Scale)'
        
    scatter3 = ax.scatter(x, y, c=error, cmap="magma", s=15, norm=norm)
    fig.colorbar(scatter3, ax=ax, label=cbar_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("3. Absolute Error Map")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

    # max_error = df_merged['error'].max()
    # mean_abs_error = df_merged['error'].mean()
    
    # max_error_point = df_merged.loc[df_merged['error'].idxmax()]

    # print("--- Error Analysis (Sequential vs Parallel) ---")
    # print(f"Total points compared: {len(df_merged)}")
    # print("-" * 40)
    # print(f"Maximum Absolute Error: {max_error:.8f}")
    # print(f"Mean Absolute Error: {mean_abs_error:.8f}")
    # print("-" * 40)
    # print("Coordinates of Maximum Error Point:")
    # print(f"x: {max_error_point['x']:.4f}, y: {max_error_point['y']:.4f}")
    # print(f"u_sequential: {max_error_point['u_seq']:.8f}")
    # print(f"u_parallel:   {max_error_point['u_par']:.8f}")

visualize_solution_error('sequential.txt', 'analytical.txt')