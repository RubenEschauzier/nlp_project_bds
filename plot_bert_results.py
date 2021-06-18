from matplotlib import pyplot as plt
def plot_bert():
    # These values were taken from the output of a google colab, this is why they are hardcoded

    validation_rmse = [0.102, 0.090, 0.090, 0.088, 0.080, 0.081, 0.095, 0.089, 0.084, 0.084]
    train_rmse = [0.147, 0.097, 0.079, 0.072, 0.068, 0.063, 0.057, 0.051, 0.046, 0.041]
    baseline_mean = 0.174
    n_epochs = list(range(1,11))

    plot_data = [validation_rmse, train_rmse]

    labels = ['Validation RMSE', 'Training RMSE']
    colors = ['r', 'b']

    # loop over data, labels and colors
    for i in range(len(plot_data)):
        plt.plot(n_epochs, plot_data[i], 'o-', color=colors[i], label=labels[i])

    plt.legend()
    plt.show()
