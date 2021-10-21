import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})
plt.style.use('default')
figure = plt.figure()
plt.rcParams.update({'errorbar.capsize': 2})

gas_source_energy = 4.77  # MeV
material_source_energy = 5.8  # MeV


def plotData(title, xAxisTitle, yAxisTitle, x, y, error_y, label):
    figure = plt.figure()
    axes_1 = figure.add_subplot(121)
    axes_1.plot(x, y, "b+", label=label)
    axes_1.errorbar(x, y, error_y, fmt="b+")
    plt.xlabel(xAxisTitle)  #
    plt.ylabel(yAxisTitle)  # edit from axes
    plt.title(title)

    # plt.savefig(title + ".png")


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
        self.differential_error = []

        # calibration data for plotting function
        self.energy = []
        self.distance = []
        self.energy_error = []

        self.I = 0

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
        axes = figure.add_subplot(111)
        axes.set_title(axesTitle)
        axes.plot(x, y, "b+", label=label)

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
        y_weights = (1 / error_y) * np.ones(np.size(y))
        fit_parameters, fit_errors = np.polyfit(x, y, degree, cov=True, w=y_weights)

        print('Fit np.polyfit of' + title)
        print("The fitting polynomial is of degree " + str(degree))
        if degree == 3:
            print('Constant  term   a = {:04.10f} +/- {:04.10f}'.format(fit_parameters[3], fit_errors[3][3]))
            print('Linear term      b = {:04.10f} +/- {:04.10f}'.format(fit_parameters[2], fit_errors[2][2]))
            print('Quadratic term   c = {:04.10f} +/- {:04.10f}'.format(fit_parameters[1], fit_errors[1][1]))
            print('Cubic term       d = {:04.10f} +/- {:04.10f}'.format(fit_parameters[0], fit_errors[0][0]))
        elif degree == 1:
            print('Gradient  m = {:04.10f} +/- {:04.10f}'.format(fit_parameters[1], fit_errors[1][1]))
            print('Intercept d = {:04.10f} +/- {:04.10f}'.format(fit_parameters[0], fit_errors[0][0]))

        return fit_parameters, fit_errors

    def returnDifferential(self, b, c, d, x, deg, err_b, err_c, err_d):
        array = []
        if deg == 2:
            function = lambda b, c, d, x: b + 2 * c * x
        elif deg == 3:
            function = lambda b, c, d, x: b + 2 * c * x + 3 * d * x ** 2
        else:
            function = lambda b, c, d, x: 0

        # errors for the differential
        errors = []

        function2 = lambda er_d, d, er_b, b, er_c, c, x: np.sqrt(er_b ** 2 + x ** 2 * er_c ** 2 + x ** 4 * er_d ** 2)

        errors = function2(err_d, d, err_b, b, err_c, c, x)
        array = function(b, c, d, x)  # gets the function for each entry of x

        self.differential = np.array(array)
        self.differential_error = np.array(errors)

    def convertChannelNumber(self, channel_number):
        energy = self.coeff[1] * channel_number + self.coeff[0]
        print("convert channel number")
        print(self.coeff)
        energy_error = np.sqrt(channel_number ** 2 * self.coeff_err[0] ** 2 + self.coeff_err[1] ** 2)
        return energy, energy_error

    def calibrationCurve(self):
        alphaEnergy = 4.77  # MeV
        zero = 0
        temp = Data("data.txt")  # calibration object
        temp.readData()
        x = temp.get_x()  # channel number
        print(x)
        y = temp.get_y()  # voltage
        err = temp.get_err()  # voltage error
        # first two points are calibration of the source energy
        alpha_point = [0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (err[0] + err[0])]
        zero_point = [x[len(x) - 1], y[len(y) - 1], err[len(err) - 1]]


        # get the two point energy calibration
        energy_values = [0, 4.77]  # MeV
        voltage = [zero_point[1], alpha_point[1]]  # the corresponding 2-point calibration points
        voltage_err = []
        # Get equation to get 1 to 1 Voltage to Energy conversion
        m = (energy_values[1] - energy_values[0]) / (voltage[1] - voltage[0])
        c = energy_values[1]
        print("fitting")
        print(m)
        print(c)

        temp_x = x
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

        fitting_coeff, fit_err = self.DataFit(temp_x, y, err, "Calibration Curve", "Channel number", "Energy", "Signal",
                                              1)
        self.coeff = [fitting_coeff[1], fitting_coeff[0]]  # mx+c
        self.coeff_err = [fit_err[0][0], fit_err[1][1]]
        print("Fitting coefficient from the second fit")
        print(self.coeff)
        print(self.coeff_err)


    def gasAnalysis(self):
        # get the calibration curve
        self.calibrationCurve()
        # convert pressure into distance
        distance = 142.3 * (np.array(self.x) / 1000)  # mm  - convert from pressure into distance
        self.x = distance  # now numpy array
        # convert channel number into energy
        energy, energy_error = self.convertChannelNumber(np.array(self.x))
        print("Energy")
        print(energy)
        print("Energy Errors")
        print(energy_error)

        self.energy = energy
        self.distance = distance
        self.energy_error = energy_error
        # fit cubic with energy vs distance
        fitting_coeff, fitting_err = self.DataFit(distance, energy, energy_error, "Energy vs Distance", "Distance/ mm",
                                                  "Energy/ MeV", "Signal", 3)
        # store the differential array - generate the array using the values from the fit
        self.returnDifferential(fitting_coeff[2], fitting_coeff[1], fitting_coeff[0], distance, 3, fitting_err[2][2],
                                fitting_err[1][1], fitting_err[0][0])  # note: distance is a numpy array
        print("differential")
        print(self.differential)
        print(self.differential_error)
        # obtain errors for the differential
        I = self.fitting_I(energy, np.array(self.differential), self.differential_error)
        print("The value of the ionisation energy is: " + str(I))

    def getChiSqrt(self, fit_y, y, ey):
        # all arrays are numpy arrays
        # returns the chi squared value
        chi_sqrt = ((y - fit_y) / ey) ** 2
        return np.sum(chi_sqrt)

    def fitting_I(self, x, y, ey):
        y = -1 * y  # make the differential -dE/dx
        function = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I))
        # declares the function we want to fit
        limit = 100
        step = 1
        chi_sqr = 101
        I = 20  # estimate using 10Z eV
        n = 0
        upper = 1000
        while chi_sqr > limit and n < upper:
            # check higher
            fit_y = function(4, 2, x, I + step)
            chi_sqr_high = self.getChiSqrt(fit_y, y, ey)
            # check lower
            fit_y = function(4, 2, x, I - step)
            chi_sqr_low = self.getChiSqrt(fit_y, y, ey)
            # adjust I
            if chi_sqr_high < chi_sqr_low:
                I += step
            n += 1

        self.I = I
        return I

    # noinspection PyTupleAssignmentBalance
    def plotting_everything(self, choice):
        # choice = [True,False,True,False,True] - array of numbers indicating which plots to do

        if (choice[0]):
            # plotting the calibration curve with fit
            # energy against channel number and the model
            self.energy_error = np.array(self.energy_error)
            y = self.energy
            x = np.array(self.distance)
            figure = plt.figure()
            axes_1 = figure.add_subplot(121)
            axes_1.plot(self.distance, self.energy, "b+")
            axes_1.errorbar(self.distance, self.energy, self.energy_error)
            plt.xlabel("Channel number")
            plt.ylabel("Energy")
            plt.title("Calibration Curve")
            y_weights = (1 / self.energy_error) * np.ones(np.size(y))
            y_errors = self.energy_error * np.ones(np.size(y))
            fit_parameters, fit_errors = np.polyfit(self.distance, self.energy, 1, cov=True, w=self.energy_error)

            y_fitted = np.polyval(fit_parameters, x)
            axes_1.plot(self.distance, y_fitted)  # fit the model onto the first plot
            axes_2 = figure.add_subplot(122)
            axes_2.set_xlabel("Channel number")
            axes_2.set_ylabel("Error")
            axes_2.errorbar(self.distance, self.energy - y_fitted, yerr=y_errors, fmt='b+')
            plt.savefig("Calibration_Curve.png")
        # end if

        if (choice[1]):
            # plotting the -differential against energy plot with fit
            x = np.array(self.energy)
            y = self.differential * -1
            y_err = np.array(self.differential_error)

            figure2 = plt.figure()
            axes_3 = figure2.add_subplot(121)
            axes_3.plot(x, y, "b+")
            axes_3.errorbar(x, y, y_err, fmt="b+")
            plt.xlabel("-dE/dx")
            plt.ylabel("E")
            plt.title("Differential Energy plot")
            y_weights = (1 / y_err) * np.ones(np.size(y))
            y_errors = self.energy_error * np.ones(np.size(y))
            fit_parameters, fit_errors = np.polyfit(x, y, 5, cov=True, w=self.energy_error)

            y_fitted = np.polyval(fit_parameters, x)
            axes_3.plot(x, y_fitted)  # fit the model onto the first plot
            axes_4 = figure2.add_subplot(122)
            axes_4.set_xlabel("-dE/dx")
            axes_4.set_ylabel("Error")
            axes_4.errorbar(x, y - y_fitted, yerr=y_errors, fmt='b+')
            plt.savefig("Differential_Plot.png")
        # end if

        if (choice[2]):
            # plotting energy against distance with fit
            # distance obtained from integrals
            x = self.distance
            x = np.array(x)
            y = np.array(self.energy)
            y_err = np.array(self.energy_error)

            figure4 = plt.figure()
            axes_7 = figure4.add_subplot(111)
            axes_7.plot(x, y, "b+")
            axes_7
            axes_7.set_xlabel("Distance")
            axes_7.set_ylabel("Energy")
            axes_7.set_title("Energy against Distance")

        # end if

        if (choice[3]):
            # plotting the model with the fit of I along with the experimental data
            x = np.array(self.energy)
            y = -1 * self.differential
            y = np.array(y)
            y_err = np.array(self.differential_error)
            function = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I))
            y_fitted = function(4, 2, x, self.I)
            figure3 = plt.figure()
            axes_5 = figure3.add_subplot(121)
            axes_5.plot(x, y, y_err, "b+")
            axes_5.set_xlabel("Energy E/ MeV")
            axes_5.set_ylabel("-dE/dx")
            axes_5.set_title("Model against the experimental data")
            y_weights = (1 / y_err) * np.ones(np.size(y))
            y_errors = self.energy_error * np.ones(np.size(y))
            axes_5.plot(x, y_fitted)  # plots the model as a line

            axes_6 = figure3.add_subplot(122)
            axes_6.set_xlabel("-dE/dx")
            axes_6.set_ylabel("Error")
            axes_6.errorbar(x, y - y_fitted, yerr=y_errors, fmt='b+')
            plt.savefig("Fitting_I_curve.png")
        # end if

def main():
    d = Data("Nitrogen.txt")  # init
    d.readData()
    d.gasAnalysis()
    #d.plotting_everything([True, True, True, True])


main()
plt.show()
