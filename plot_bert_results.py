from matplotlib import pyplot as plt
def plot_bert():
    # These values were taken from the output of a google colab, this is why they are hardcoded

    validation_rmse = [0.260, 0.178, 0.173, 0.173, 0.173, 0.174, 0.173, 0.176, 0.173, 0.176]
    train_rmse = [0.336, 0.271, 0.225, 0.188, 0.176, 0.175, 0.176, 0.176, 0.176, 0.176]
    n_epochs = list(range(1,11))

    plot_data = [validation_rmse, train_rmse]

    labels = ['Validation RMSE', 'Training RMSE']
    colors = ['r', 'b']

    # loop over data, labels and colors
    for i in range(len(plot_data)):
        plt.plot(n_epochs, plot_data[i], 'o-', color=colors[i], label=labels[i])

    plt.legend()
    plt.show()
