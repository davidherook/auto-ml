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

def plot_nn_histories(histories, save_to=None):
    # TODO
    # Check what metrics are available for regression - can we substitute R2 for acc?
    metrics=['loss', 'accuracy', 'val_accuracy', 'val_loss']

    avg_loss = np.mean([history.history['loss'][-1] for history in histories])
    avg_val_loss = np.mean([history.history['val_loss'][-1] for history in histories])
    avg_acc = np.mean([history.history['accuracy'][-1] for history in histories])
    avg_val_acc = np.mean([history.history['val_accuracy'][-1] for history in histories])

    coords = {'loss': [0, 0],
        'accuracy': [0, 1],
        'val_loss': [1, 0],
        'val_accuracy': [1, 1]}

    values = {'loss': avg_loss,
        'accuracy': avg_acc,
        'val_loss': avg_val_loss,
        'val_accuracy': avg_val_acc}

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    for metric in metrics:
        ax = axes[coords[metric][0], coords[metric][1]]
        for history in histories:
            ax.plot(history.history[metric])
        ax.set_title(metric + ' (avg={0:.{1}f})'.format(values[metric], 2), size=12)
        ax.tick_params(which='major', width=0.8, labelsize=8)

    if save_to is not None:
        fig.savefig(save_to)
    plt.close()

            
