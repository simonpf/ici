import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def plot_sigma():
    x = np.linspace(0.0, 2.0*np.pi, 101)
    sigma = 0.5 * np.sin(0.5 * x)
    plt.plot(x, sigma)

def scatter_plot():
    x_train, y_train, _, _, _, var = load_data()
    norm = matplotlib.colors.Normalize(vmin=np.min(y_train[:10000]), vmax=np.max(y_train[:10000]))
    cmap  = matplotlib.cm.ScalarMappable(norm=norm, cmap="plasma")
    x = np.linspace(0.0, 2.0 * np.pi, 101)
    y = np.sin(x)
    s = 0.1 + np.sin(0.5 * x)
    plt.scatter(x_train[:10000],  y_train[:10000], s=1, c=cmap.to_rgba(y_train))
    plt.plot(x, y, c='k', lw=1)
    plt.plot(x, y + 1.96 * s, c='k', ls="--")
    plt.plot(x, y - 1.96 * s, c='k', ls="--")
    plt.xlim([0, 2.0 * np.pi])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

def load_data():
    x_train = np.load("/home/simonpf/projects/ici/data/sets/toy/x_train.npy")
    y_train = np.load("/home/simonpf/projects/ici/data/sets/toy/y_train.npy")
    x_test = np.load("/home/simonpf/projects/ici/data/sets/toy/x_test.npy")
    y_test = np.load("/home/simonpf/projects/ici/data/sets/toy/y_test.npy")
    mu_test = np.load("/home/simonpf/projects/ici/data/sets/toy/mu_test.npy")
    var_test = np.load("/home/simonpf/projects/ici/data/sets/toy/var_test.npy")
    return (x_train, y_train, x_test, y_test, mu_test, var_test)

def generate_data():
    x_train  = np.reshape(np.random.uniform(0.0, 2.0 * np.pi, (1e5)), (-1,1))
    y_train  = np.random.normal(np.sin(x_train), 0.1 + np.sin(0.5 * x_train)).ravel()
    x_test   = np.reshape(np.random.uniform(0.0, 2.0 * np.pi, (1e4)), (-1, 1))
    y_test   = np.random.normal(np.sin(x_test), 0.1 + np.sin(0.5 * x_test)).ravel()
    mu_test  = np.sin(x_test)
    var_test = 0.1 + np.sin(0.5 * x_test)

    np.save("/home/simonpf/projects/ici/data/sets/toy/x_train.npy", x_train)
    np.save("/home/simonpf/projects/ici/data/sets/toy/y_train.npy", y_train)
    np.save("/home/simonpf/projects/ici/data/sets/toy/x_test.npy", x_test)
    np.save("/home/simonpf/projects/ici/data/sets/toy/y_test.npy", y_test)
    np.save("/home/simonpf/projects/ici/data/sets/toy/mu_test.npy", mu_test)
    np.save("/home/simonpf/projects/ici/data/sets/toy/var_test.npy", var_test)

def plot_results(x_test, y_pred):
    x = np.linspace(0.0, 2.0 * np.pi, 101)
    y = np.sin(x)
    s = 0.1 + np.sin(0.5 * x)
    plt.plot(x, y, c='k', lw=1)
    plt.scatter(x_test, y_pred, c='r', s=1)

plt.show()
