import matplotlib.pyplot as plt
import numpy as np

def plot_nn_history(history, show=True, save_to=None):
    start = 1
    loss = history.history['loss'][start:]
    val_loss = history.history['val_loss'][start:]
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()

def plot_nn_accuracy(history, show=True, save_to=None):
    start = 1
    acc = history.history['accuracy'][start:]
    val_acc = history.history['val_accuracy'][start:]
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Val Accuracy'], loc='upper left')
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()

def plot_pred_vs_actual(y_true, y_pred, title="Actual vs. Predicted", save_to=None):
    plt.scatter(y_pred, y_true)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
    plt.clf()

def plot_residual(y_true, y_pred, title="Residuals", save_to=None):
    resid = y_true - y_pred
    plt.scatter(y_pred, resid)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
    plt.clf()

def plot_nn_histories(histories, metrics, save_to=None):
    """
    Plot the loss and metrics from training. These are defined in the neural net architecture in 
    config.yaml. The metric plotted is the last one listed if more than one is defined.

    Parameters:
        histories: a list of ... objects
        metrics: the list of neural net metrics from config.yaml
        save_to: the file path to save plot to
    """

    metric = metrics[-1]
    val_metric = f'val_{metric}'
    loss = 'loss'
    val_loss = 'val_loss'
    metrics = [loss, metric, val_loss, val_metric]

    avg_loss =     np.mean( [history.history[loss][-1] for history in histories] )
    avg_val_loss = np.mean( [history.history[val_loss][-1] for history in histories] )
    avg_acc =      np.mean( [history.history[metric][-1] for history in histories] )
    avg_val_acc =  np.mean( [history.history[val_metric][-1] for history in histories] )

    coords = {
        loss: [0,0],
        metric: [0,1],
        val_loss: [1,0],
        val_metric: [1,1]}

    values = {
        loss: avg_loss,
        metric: avg_acc,
        val_loss: avg_val_loss,
        val_metric: avg_val_acc}

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    for m in metrics:
        ax = axes[coords[m][0], coords[m][1]]
        for history in histories:
            ax.plot(history.history[m])
        ax.set_title(m + ' (avg={0:.{1}f})'.format(values[m], 2), size=12)
        ax.tick_params(which='major', width=0.8, labelsize=8)

    if save_to is not None:
        fig.savefig(save_to)
    plt.close()

            
