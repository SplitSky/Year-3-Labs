import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self, filename):
        self.x = []
        self.y = []
        self.filename = filename
        self.coeff = []  # m and c
        self.coeff_err = []  # errors in m and c for calibration curve

        self.cal_x = []
        self.cal_y = []
        self.cal_err = []

        self.differential = []

    # end def

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_err(self):
        return self.err

    def plotCustom(self):
        # get data
        print(self.x)
        print(self.y)
        plt.plot(np.array(self.x), np.array(self.y), "b+")

    def plotCustom2(self, x, y, axesTitle, label):
        figure = plt.figure()
        axes_1 = figure.add_subplot(111)
        axes_1.set_title(axesTitle)
        axes_1.plot(x, y, "b+", label=label)

    def readData(self):
        array = [[], [], []]
        channel = []
        pressure = []
        err = []

        # file open
        file = open(self.filename, "r")
        content = file.readlines()
        temp2 = 0
        divider = len(content) / 3
        print("divider" + str(divider))
        for counter in range(0, len(content)):
            temp = content[counter]
            if (counter >= 0) and (counter < divider):
                # x values
                channel.append(float(content[counter]))
            elif (counter >= divider) and (counter < divider * 2):
                # y values
                pressure.append(float(content[counter]))
            elif (counter >= divider * 2) and (counter <= divider * 3):
                # error values
                err.append(float(content[counter]))
            else:
                # error control
                print("Error entry")

        self.x = channel
        self.y = pressure
        self.err = err
        # Data loaded
        print("Data loaded.")
        file.close()

    def DataFit(self, x, y, error_y, title, xAxisTitle, yAxisTitle, label, degree):
        figure = plt.figure()
        axes_1 = figure.add_subplot(121)
        axes_1.plot(x, y, "b+", label=label)
        axes_1.errorbar(x, y, error_y, fmt="b+")
        plt.xlabel(xAxisTitle)  #
        plt.ylabel(yAxisTitle)  # edit from axes
        plt.title(title)
        y_weights = (1 / error_y) * np.ones(np.size(y))
        y_errors = error_y * np.ones(np.size(y))
        fit_parameters, fit_errors = np.polyfit(x, y, degree, cov=True, w=y_weights)

        y_fitted = np.polyval(fit_parameters, x)
        axes_1.plot(x, y_fitted)
        axes_2 = figure.add_subplot(122)
        axes_2.set_xlabel(xAxisTitle)
        axes_2.set_ylabel(yAxisTitle)
        axes_2.errorbar(x, y - y_fitted, yerr=y_errors, fmt='b+')

        # plt.savefig(title + ".png")
        print("fit parameters" + str(fit_parameters))
        # a + bx + cx^2 + dx^3 = y
        return fit_parameters, fit_errors

    def cubicFitData(self, title, xAxisTitle, yAxisTitle, label):
        x = self.x
        y = self.y
        error_y = self.err
        error_y = np.array(error_y)

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

        # plt.savefig(title + ".png")
        print(fit_parameters[0])
        print(fit_parameters[1])
        print(fit_parameters[2])
        print(fit_parameters[3])
        # a + bx + cx^2 + dx^3 = y

    def returnDifferential(self, b, c, d, x, deg):

        array = []
        if deg == 2:
            function = lambda b, c, d, x: b + 2 * c * x
        elif deg == 3:
            function = lambda b, c, d, x: b + 2 * c * x + 3 * d * x ** 2
        else:
            function = lambda b, c, d, x: 0

        for entry in x:
            array.append(function(b, c, d, x))
        return np.array(array)

    def convertChannelNumber(self, channel_number):
        energy = self.coeff[0] * channel_number + self.coeff[1]
        energy_error = np.sqrt(channel_number ** 2 * self.coeff_err[0] ** 2 + self.coeff_err[1] ** 2)
        return [energy, energy_error]

    def error_prop_Energy(self, x, m, err_m, err_c, err_x):
        error = m ** 2 * err_x ** 2 + err_c ** 2 + x ** 2 * err_m ** 2
        error = np.sqrt(error)
        return error

    def calibrationCurve(self):
        alphaEnergy = 4.77  # MeV
        zero = 0
        temp = Data("data.txt")  # calibration object
        temp.readData()
        x = temp.get_x()  # channel number
        y = temp.get_y()  # voltage
        err = temp.get_err()  # voltage error
        # first two points are calibration of the source energy
        alpha_point = [0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (err[0] + err[0])]
        zero_point = [x[len(x) - 1], y[len(y) - 1], err[len(err) - 1]]

        print("Debug 1")
        print(len(x))
        print(len(y))
        print(len(err))
        print("Debug end")

        # get the two point energy calibration
        energy_values = [0, 4.77]  # MeV
        voltage = [zero_point[1], alpha_point[1]]  # the corresponding 2-point calibration points
        voltage_err = []
        # Get equation to get 1 to 1 Voltage to Energy conversion
        m = (energy_values[1] - energy_values[0]) / (voltage[1] - voltage[0])
        c = energy_values[1]

        temp_x = x
        # print(temp_x)
        energy_values = np.array(temp_x) * m + c  # gets the energy values at the same values as the voltage
        # finds the energy values to which each voltage corresponds

        # voltage to energy conversion finished
        # x value: channel number
        # y value: energy
        # err: on energy

        err = np.array(err)

        err = np.sqrt(err ** 2 * m ** 2)
        y = energy_values
        x = np.array(x)
        print(y.size)
        print(err.size)
        print(x.size)

        fitting_coeff, fit_err = self.DataFit(temp_x, y, err, "Calibration Curve", "Channel number", "Energy", "Signal",
                                              1)
        self.coeff = [fitting_coeff[0], fitting_coeff[1]]  # mx+c
        self.coeff_err = [fit_err[0][0], fit_err[1][1]]

    def gasAnalysis(self):
        # get the calibration curve
        self.calibrationCurve()
        # convert pressure into distance
        distance = 142.3 * (np.array(self.x) / 1000)  # mm
        self.x = distance  # now numpy array
        # convert channel number into energy
        energy, energy_error = self.convertChannelNumber(np.array(self.x))
        # fit cubic with energy vs distance
        fitting_coeff, fitting_err = self.DataFit(distance, energy, energy_error, "Energy vs Distance", "Distance/ mm",
                                                  "Energy/ MeV", "Signal", 2)
        # store the differential array
        self.differential = self.returnDifferential(fitting_coeff[2], fitting_coeff[1], fitting_coeff[0], distance,3)
        # plot diff. on y axis and energy on x-axis
        self.plotCustom2(energy, self.differential, "dE/dx vs Energy: Bragg Curve", "Differential")
        # obtain errors for the differential


def main():
    # 1. load the data for the selected sample
    # 2. fit the polynomial
    # 3. plot the differential against energy - Bragg curve
    # 4. integrate under the curve using an integrator
    # 5.

    filenames = ["data.txt", "Helium.txt", "Argon.txt", "Nitrogen.txt", "Helium2.txt"]
    # Gas analysis

    print("Which data set?")
    print("1. Calibration for gas")
    print("2. Helium")
    print("3. Argon")
    print("4. Nitrogen")
    print("5. Helium 2 electric boogaloo")
    indicator = True
    choice = 0
    while indicator:
        try:
            indicator = False
            choice = int(input("Choice: "))  # choice is index+1
        except:
            print("Try again")

    if choice in [2, 3, 4, 5]:
        # gas analysis
        print(filenames[choice - 1] + str(" File is being loaded"))
        d = Data(filenames[choice - 1])
        d.readData()
        print("Gas Analysis")
        d.gasAnalysis()
        d.plotCustom()

    elif choice == 2:
        print("Analysing Helium data set 1")
        print(filenames[choice - 1] + str(" File is being loaded"))
        d = Data(filenames[choice - 1])
        d.readData()
        print("Gas Analysis")
        d.gasAnalysis()
        d.plotCustom()

    elif choice == 3:
        print("Analysing Argon data set")
        print(filenames[choice - 1] + str(" File is being loaded"))
        d = Data(filenames[choice - 1])
        d.readData()
        print("Gas Analysis")
        d.gasAnalysis()
        d.plotCustom()

    elif choice == 4:
        print("Analysing Nitrogen data set")
        print(filenames[choice - 1] + str(" File is being loaded"))
        d = Data(filenames[choice - 1])
        d.readData()
        print("Gas Analysis")
        d.gasAnalysis()
        d.plotCustom()

    elif choice == 5:
        print("Analysing Helium data set 2")
        print(filenames[choice - 1] + str(" File is being loaded"))
        d = Data(filenames[choice - 1])
        d.readData()
        print("Gas Analysis")
        d.gasAnalysis()
        d.plotCustom()


    elif (choice == 0):
        # gas calibration
        d = Data(filenames[choice - 1])
        d.readData()
        print("Calibration Analysis")
    elif (choice > 5):  # material analysis
        d = Data(filenames[choice - 1])
        print("Metal Analysis")
    else:
        print("error")


main()
plt.show()
