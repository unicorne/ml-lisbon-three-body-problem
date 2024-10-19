
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

def plot_y_yhat(y_test: pd.DataFrame, y_pred: pd.DataFrame, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'] #y_test.columns.tolist()  # Get column names as labels
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
    else:
        idx = np.arange(len(y_test))

    # Calculate mean RMSE
    #RMSE = np.sqrt(np.mean((y_test.values - y_pred.values) ** 2))
    mean_RMSE = np.mean(np.sqrt(np.mean((y_test.values - y_pred.values) ** 2, axis=0)))

    # Create the figure with subplots
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'{plot_title} (Mean RMSE: {mean_RMSE:.4f})', fontsize=16, weight='bold')

    for i, col in enumerate(labels):
        # Extract the test and predicted values for the current column
        y_test_col = y_test.iloc[idx, i]
        y_pred_col = y_pred.iloc[idx, i]

        # Determine min and max for the current column
        x0 = np.min(y_test_col)
        x1 = np.max(y_test_col)
        
        # Create subplot
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_test_col, y_pred_col, edgecolors='b', facecolors='none')
        plt.xlabel(f'True {col}', fontsize=12)
        plt.ylabel(f'Predicted {col}', fontsize=12)
        plt.plot([x0, x1], [x0, x1], color='red', linestyle='--', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('square')
        # fix x and y axis
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        
        # Calculate and display RMSE for each variable
        rmse_individual = np.sqrt(np.mean((y_test_col - y_pred_col) ** 2))
        plt.title(f'{col}: RMSE = {rmse_individual:.4f}', fontsize=10)

    # Adjust layout for clarity
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    plt.savefig(f'images/{plot_title}.png')

    # Show the plot
    plt.show()

def create_submission(model, submission_name, x_test_real, x_columns_train, y_columns_train, x_columns_submissions):
    for col in x_columns_train:
        if col not in x_test_real.columns:
            x_test_real[col] = 0

    y_test_real_pred = model.predict(x_test_real[x_columns_train])

    # create dataframe
    y_test_real_pred = pd.DataFrame(y_test_real_pred, columns=y_columns_train)
    # add ID column
    y_test_real_pred = y_test_real_pred[x_columns_submissions]
    y_test_real_pred['Id'] = np.arange(0, len(y_test_real_pred))
    submission_columns = ["Id","x_1","y_1","x_2","y_2","x_3","y_3"]
    y_test_real_pred = y_test_real_pred[submission_columns]
    y_test_real_pred.to_csv(f"submissions/{submission_name}", index=False)
    return y_test_real_pred

def create_random_submission(submission_name):
    submission_columns = ["Id","x_1","y_1","x_2","y_2","x_3","y_3"]
    y_test_real_pred = np.random.rand(1041621, 6)
    # add random sign 
    y_test_real_pred = y_test_real_pred * np.random.choice([-1, 1], size=y_test_real_pred.shape)
    # add continuning id
    id_values = np.arange(0, len(y_test_real_pred))
    y_test_real_pred = np.column_stack((id_values, y_test_real_pred))
    y_test_real_pred = pd.DataFrame(y_test_real_pred, columns=submission_columns)

    # add ID column
    y_test_real_pred['Id'] = np.arange(0, len(y_test_real_pred))
    y_test_real_pred = y_test_real_pred[submission_columns]
    y_test_real_pred.to_csv(f"submissions/{submission_name}", index=False)
    return y_test_real_pred

    
def calculate_rmse_error_over_time(x_val, y_val, model, x_columns_train, y_columns_train):
    rmse_errors=[]
    example_t_values = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0,2.0,3.0]
    for t_value in example_t_values:
        # get example
        x_example = x_val[x_val["t"]<=t_value]
        y_example = y_val[x_val["t"]<=t_value][y_columns_train]
        # predict
        y_pred_example = model.predict(x_example[x_columns_train])

        # calculate rmse
        rmse = np.sqrt(np.mean((y_example - y_pred_example) ** 2))

        rmse_errors.append(rmse)
    return example_t_values, rmse_errors

def plot_trajectories(df, max_time):
    """
    Plots the trajectories of three bodies in 2D space over time.

    Parameters:
    - df: pandas DataFrame containing the positions with columns ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    - time: int, the time index up to which the trajectories should be plotted
    """

    # Limit the data up to the specified time
    #df = df.iloc[:time+1]
    df = df[df["t"] <= max_time]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['red', 'blue', 'green']  # Colors for the three bodies
    labels = ['Body 1', 'Body 2', 'Body 3']

    for i, color in enumerate(colors, start=1):
        x = df[f'x_{i}'].values
        y = df[f'y_{i}'].values

        # Create points and segments for LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a colormap that goes from the solid color to white
        cmap = mcolors.LinearSegmentedColormap.from_list('', [color, 'white'])
        norm = plt.Normalize(0, 1)

        # Create LineCollection with varying color
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.linspace(0, 1, len(segments)))
        lc.set_linewidth(2)

        # Add the LineCollection to the plot
        ax.add_collection(lc)

        # Plot the start position
        ax.scatter(x[0], y[0], color=color, edgecolor='black', s=100, label=f'{labels[i-1]} Start')

        # Plot the end position
        ax.scatter(x[-1], y[-1], color=color, edgecolor='black', marker='X', s=100, label=f'{labels[i-1]} End')

    # Set plot limits
    all_x = np.concatenate([df[f'x_{i}'].values for i in range(1, 4)])
    all_y = np.concatenate([df[f'y_{i}'].values for i in range(1, 4)])
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)

    # Add grid, legend, and labels
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Trajectories of Three Bodies until t={max_time}s')

    #save fig
    plt.savefig(f'images/trajectories_t{max_time}.png')

    # Display the plot
    plt.show()
