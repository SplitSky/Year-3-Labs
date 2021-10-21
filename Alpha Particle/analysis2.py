import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})
plt.style.use('default')
figure = plt.figure()
plt.rcParams.update({'errorbar.capsize': 2})

gas_source_energy = 4.77  # MeV
material_source_energy = 5.8  # MeV


def get_choice():
    print("Enter the data set number:")
    print("1. Nitrogen")
    print("2. Helium 1")
    print("3. Helium 2")
    print("4. Argon")
    print("5. Nickel")
    print("6. Aluminium")
    choice = input("Enter the name: ")
    file_names = ["Nitrogen.txt", "Helium.txt", "Helium2.txt", "Argon.txt", "Nickel.txt", "Aluminium.txt"]
    if int(choice) - 1 in [0, 1, 2, 3]:
        type = True
    else:
        type = False

    return file_names[int(choice) - 1], type


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
        self.readData()

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_err(self):
        return self.err

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

    def calibrationCurve(self, type):
        fit_param = []
        fit_err = []
        if type:
            # gasses
            V = [42.4, 50, 64, 30.8, 7.92, 6.8, 55.6, 49.6, 42.8, 34.8, 27.4, 13.8, 64, 52.8, 46.8, 38.8, 31, 0.304]
            E = [4.77, 0]
            ch = [439.7, 459.8, 440.45, 220.5, 56.14, 44.45, 399.28, 350.36, 300.7, 249.5, 197.62, 100.07, 458.83,
                  378.18, 333.96, 271.38, 221.4, 0]
            Volt_err = [3, 3, 1, 1, 1, 0.4, 1, 1, 1, 0.5, 0.5, 0.4, 1, 1, 1, 1, 1, 0.2]
            dif_V = 0.5 * (42.4 + 50) - 0.304
            dif_E = 4.77  # MeV
            m_1 = dif_E / dif_V
            c_1 = m_1 * 0.304 * -1
            # relate voltage to energy
            Vin_Eout = lambda V: m_1 * V + c_1

            # relate channel number to voltage
            def Chin_Vout(ch):
                y_weights = Volt_err
                fit_parameters, fit_errors = np.polyfit(ch, V, 1, cov=True, w=y_weights)
                return fit_parameters[0] * ch + fit_parameters[1]
            def function(ch):
                return Vin_Eout(Chin_Vout(ch))
        else:
            # materials
            V = [88.8, 88.8, 104, 10.3, 5.72, 95.2, 85.6, 77.6, 68, 57.6, 47.6, 37, 26.6, 15.9, 5.32, 10.7, 21.4, 31.6,
                 42.4, 52.4, 62.4, 72.4]
            E = [5.8, 0]
            ch = [880.62, 880.52, 880.64, 84.53, 47.86, 806.3, 729.72, 652.29, 570.34, 486.16, 400.41, 312.37, 224.12,
                  134.76, 46.72, 89.56, 178.66, 267.57, 355.56, 442.22, 527.39, 610.54]
            Volt_err = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                        0.35, 0.35, 0.35, 0.35, 0.35, 0.35]

            dif_V = 88.8 - 0.304
            dif_E = 5.8  # MeV
            m_1 = dif_E / dif_V
            c_1 = m_1 * 0.304 * -1  # assuming same systematic shift in the zero value

            # m_1 = 5.8/880.62
            # y = np.array(ch)
            # y = ch * k # scale with k

            # relate voltage to energy
            def Vin_Eout(V):  # assuming the intercept is at (0,0)
                return m_1 * V

            # relate channel number to voltage
            def Chin_Vout(ch):
                y_weights = Volt_err
                fit_parameters, fit_errors = np.polyfit(ch, V, 1, cov=True, w=y_weights)
                return fit_parameters[0] * ch + fit_parameters[1]

            def function(ch):
                return Vin_Eout(Chin_Vout(ch))
        # end if

        x = ch
        y = function(x)
        Volt_err = np.array(Volt_err)
        y_err = np.sqrt(m_1 ** 2 * Volt_err ** 2) ## propagates the errors from V onto the energy
        fit_param, fit_err = np.polyfit(x, y, 1, weight=y_err) # final fit with the proper weighting
        print(fit_param)
        print(fit_err)

        self.coeff = fit_param
        self.coeff_err = []
        self.coeff_err.append(fit_err[0][0])
        self.coeff_err.append(fit_err[1][1])
        # assigns the value within the object from the correct calibration

    def materialAnalysis(self):

        ### placeholder
        a = 1

    def gasAnalysis(self):
        # get the calibration curve
        self.calibrationCurve(type)
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
    file_name, type = get_choice()
    d = Data(file_name)  # init
    if type:
        d.gasAnalysis()
    else:
        d.materialAnalysis()

    # d.plotting_everything([True, True, True, True])


main()
plt.show()
