import matplotlib.pyplot as plt

def plot_nn_history(history, show=True, save_to=None):
    start = 1
    loss = history.history['loss'][start:]
    val_loss = history.history['val_loss'][start:]
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
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