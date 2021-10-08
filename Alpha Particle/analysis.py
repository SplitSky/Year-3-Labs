import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self, filename):
        self.x = []
        self.y = []
        self.filename = filename
    # end def

    def plotCustom(self):
        # get data
        plt.plot(self.x, self.y, "b+")

    # end def

    def plotCustom2(self, x, y, axesTitle, label):
        figure = plt.figure()
        axes_1 = figure.add_subplot(111)
        axes_1.set_title(axesTitle)
        axes_1.plot(x, y, "b+", label=label)

    def readData(self):
        array = [[], [], []]
        channel = []
        voltage = []
        err = []

        # file open
        file = open(self.filename, "r")
        content = file.readlines()
        temp2 = 0
        divider = len(content)/3
        for counter in range(0,len(content)):
            temp = content[counter]
            if (counter >= 0) and (counter < divider):
                #x values
                channel.append(content[counter])
            elif (counter >= divider) and (counter < divider*2):
                #y values
                voltage.append(content[counter])
            elif (counter >= divider*2) and (counter <= divider*3):
                #error values
                err.append(content[counter])
            else:
                # error control
                print("Error entry")


        self.x = channel
        self.y = voltage
        self.err = err
        # Data loaded
        print("Data loaded.")
        file.close()
        print(self.x[0])
    #end def

    def cubicFitData(self, title, xAxisTitle, yAxisTitle, label):
        x = self.x
        y = self.y
        error_y = self.err

        figure = plt.figure()
        axes_1 = figure.add_subplot(121)
        axes_1.plot(x, y, "b+", label=label)
        axes_1.errorbar(x, y, error_y, fmt="b+")
        plt.xlabel(xAxisTitle)  #
        plt.ylabel(yAxisTitle)  # edit from axes
        plt.title(title)
        y_weights = (1 / error_y) * np.ones(np.size(y))
        y_errors = error_y * np.ones(np.size(y))
        fit_parameters, fit_errors = np.polyfit(x, y, 3, cov=True, w=y_weights)

        y_fitted = np.polyval(fit_parameters, x)
        axes_1.plot(x, y_fitted)
        axes_2 = figure.add_subplot(122)
        axes_2.set_xlabel(xAxisTitle)
        axes_2.set_ylabel(yAxisTitle)
        axes_2.errorbar(x, y - y_fitted, yerr=y_errors, fmt='b+')

        plt.savefig(title + ".png")
        print(fit_parameters[0])
        print(fit_parameters[1])
        print(fit_parameters[2])
        print(fit_parameters[3])
        # a + bx + cx^2 + dx^3 = y

    def returnDifferential(self,b,c,d,x):
        temp = np.array(x)
        return b+ 2 * c * temp + 3 * d * temp^2 # return a numpy array



    ### integrator

    ### iteration fitter???


def main():
    # 1. load the data for the selected sample
    # 2. fit the polynomial
    # 3. plot the differential against energy - Bragg curve
    # 4. integrate under the curve using an integrator
    # 5.
    d = Data("data.txt")
    d.readData()
    d.cubicFitData("testing", "x-axis","y-axis", "line")

main()
plt.show()