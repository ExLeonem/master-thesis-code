import matplotlib.pyplot as plt
import seaborn as sns


def loss_in_steps(losses, step_size, num_plots, figsize=(22, 10), num_columns=5, from_epoch=None, to_epoch=None):
     """
        Plots the losses of a training procedure.

        Parameters:
            losses (pandas.DataFrame): The losses as a dataframe.
            step_size (int): how many iterations to include into one plot
            num_plots (int): the number of total plots.
            figsize (tuple): The size of the plots
    """
    
    num_rows = math.ceil(num_plots/num_columns)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)    
    
    column_idx = 0
    range_start = 0
    row_idx = 0
    for i in range(num_plots):
        
        range_end = range_start + step_size
        
        data = losses.T[range_start:range_end].T
        
        if to_epoch is not None and from_epoch is not None:
            data = data[from_epoch:to_epoch]
        
        elif to_epoch is not None:
            data = data[:to_epoch]
        
        elif from_epoch is not None:
            data = data[from_epoch:]
        
        sns.lineplot(data=data, ax=axes[row_idx, column_idx])
        axes[row_idx, column_idx].set_ylabel("Loss")
        axes[row_idx, column_idx].set_xlabel("Epoch")
        axes[row_idx, column_idx].legend(title="Iteration")
        
        
        # Update range of data to plot
        range_start += step_size
        column_idx += 1
        
        # Update row index
        if column_idx % num_columns == 0:
            row_idx += 1
            column_idx = 0