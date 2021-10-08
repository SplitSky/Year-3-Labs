import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self, x, y, err):
        self.x = x
        self.y = y
        self.error = err

        #self.x_div = np.abs(self.x[0] - self.x[1])
        #print("The x division is: " + str(self.x_div))
        #self.signals = []

    def linearPlotData(self, title, xAxisTitle, yAxisTitle, x, y, error_y, label):
        figure = plt.figure()
        axes_1 = figure.add_subplot(121)
        axes_1.plot(x, y, "b+", label=label)
        axes_1.errorbar(x, y, error_y, fmt="b+")
        plt.xlabel(xAxisTitle)  #
        plt.ylabel(yAxisTitle)  # edit from axes
        plt.title(title)
        y_weights = (1 / error_y) * np.ones(np.size(y))
        y_errors = error_y * np.ones(np.size(y))
        fit_parameters, fit_errors = np.polyfit(x, y, 1, cov=True, w=y_weights)

        y_fitted = np.polyval(fit_parameters, x)
        axes_1.plot(x, y_fitted)
        axes_2 = figure.add_subplot(122)
        axes_2.set_xlabel(xAxisTitle)
        axes_2.set_ylabel(yAxisTitle)
        axes_2.errorbar(x, y - y_fitted, yerr=y_errors, fmt='b+')

        plt.savefig(title + ".png")

    def plotCustom(self):
        # get data
        plt.plot(self.x, self.y, "b+")

    def plotCustom2(self, x, y, axesTitle, label):
        figure = plt.figure()
        axes_1 = figure.add_subplot(111)
        axes_1.set_title(axesTitle)
        axes_1.plot(x, y, "b+", label=label)

    def readData(self):
        array = [[], [], []]
        channel = [439.7,
                   440.45,
                   220.5,
                   56.14,
                   44.45,
                   399.28,
                   350.36,
                   300.7,
                   249.5,
                   197.62,
                   100.07,
                   0,
                   458.83,
                   378.18,
                   333.96,
                   271.38,
                   221.4,
                   459.8]
        voltage = [42.4,
                   48,
                   22.8,
                   5.92,
                   5.4,
                   40.4,
                   36,
                   31.2,
                   25.6,
                   20.2,
                   10.2,
                   0.212,
                   46.4,
                   38.8,
                   34,
                   28.4,
                   22.8,
                   47.6]
        err = [3,
               1,
               1,
               1,
               0.4,
               1,
               1,
               1,
               0.5,
               0.5,
               0.4,
               0.2,
               1,
               1,
               1,
               1,
               0.5,
               3]

        self.x = channel
        self.y = voltage
        self.err = err
        # Data loaded


def main():

    print(file)
    choice = int(input("Enter number: "))
    d = Data()
    d.plotCustom()


main()
plt.show()
