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
    def __init__(self, set):
        self.x = []
        self.y = []
        self.set = set
        self.coeff = []
        self.differential = []
        self.differential_error = []
        self.fitting_data = []

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_err(self):
        return self.err

    def readData(self):
        """
        Argon      = 0
        Nitrogen   = 1
        Helium 1   = 2
        Helium 2   = 3
        Nickel     = 4
        Aluminium  = 5
        """

        argon = [[7.00116E-4, 4.70858E-6, 5.08E-8],
                 [0.00238, 4.56522E-6, 5.08E-8],
                 [0.0036, 4.46179E-6, 5.08E-8],
                 [0.00501, 4.31305E-6, 5.08E-8],
                 [0.00638, 4.16806E-6, 5.08E-8],
                 [0.00795, 4.01079E-6, 5.08E-8],
                 [0.00928, 3.85209E-6, 5.08E-8],
                 [0.01077, 3.70172E-6, 5.08E-8],
                 [0.01202, 3.55663E-6, 5.08E-8],
                 [0.01363, 3.37162E-6, 5.08E-8],
                 [0.0148, 3.22958E-6, 5.08E-8],
                 [0.01636, 3.01633E-6, 5.08E-8],
                 [0.01793, 2.80855E-6, 5.08E-8],
                 [0.01921, 2.55892E-6, 5.08E-8],
                 [0.02078, 2.38224E-6, 5.08E-8],
                 [0.0222, 2.14033E-6, 5.08E-8],
                 [0.02334, 1.87048E-6, 5.08E-8],
                 [0.02504, 1.5926E-6, 5.08E-8],
                 [0.02647, 1.21303E-6, 5.08E-8],
                 [0.02789, 9.68781E-7, 5.08E-8],
                 [0.02917, 6.53414E-7, 5.08E-8],
                 [0.02006, 2.42125E-6, 5.08E-8],
                 [0.0212, 2.22415E-6, 5.08E-8],
                 [0.02263, 1.96416E-6, 5.08E-8],
                 [0.02419, 1.66149E-6, 5.08E-8],
                 [0.02547, 1.36248E-6, 5.08E-8],
                 [0.02704, 1.08532E-6, 5.08E-8],
                 [0.02832, 7.83869E-7, 5.08E-8],
                 [0.03003, 3.55726E-7, 5.08E-8]]

        helium = [[8.538E-4, 4.77177E-6, 5.08E-8],
                  [0.0025, 4.7167E-6, 5.08E-8],
                  [0.00388, 4.70573E-6, 5.08E-8],
                  [0.00499, 4.68673E-6, 5.08E-8],
                  [0.00632, 4.66275E-6, 5.08E-8],
                  [0.00795, 4.59996E-6, 5.08E-8],
                  [0.00928, 4.55628E-6, 5.08E-8],
                  [0.01073, 4.46148E-6, 5.08E-8],
                  [0.01221, 4.35165E-6, 5.08E-8],
                  [0.0138, 4.19844E-6, 5.08E-8],
                  [0.01508, 4.08546E-6, 5.08E-8],
                  [0.01651, 3.92839E-6, 5.08E-8],
                  [0.01779, 3.75679E-6, 5.08E-8],
                  [0.01921, 3.58508E-6, 5.08E-8],
                  [0.02078, 3.37955E-6, 5.08E-8],
                  [0.0222, 3.23527E-6, 5.08E-8],
                  [0.02391, 3.00972E-6, 5.08E-8],
                  [0.02504, 2.86931E-6, 5.08E-8],
                  [0.02647, 2.66936E-6, 5.08E-8],
                  [0.02775, 2.36365E-6, 5.08E-8],
                  [0.02974, 2.151E-6, 5.08E-8],
                  [0.03273, 1.6178E-6, 5.08E-8]]

        helium2 = [[8.538E-4, 4.77512E-6, 5.08E-8],
                   [0.00215, 4.75927E-6, 5.08E-8],
                   [0.00406, 4.71498E-6, 5.08E-8],
                   [0.00515, 4.70187E-6, 5.08E-8],
                   [0.00663, 4.67098E-6, 5.08E-8],
                   [0.0078, 4.65076E-6, 5.08E-8],
                   [0.00936, 4.60952E-6, 5.08E-8],
                   [0.01083, 4.5892E-6, 5.08E-8],
                   [0.01214, 4.57202E-6, 5.08E-8],
                   [0.01365, 4.5449E-6, 5.08E-8],
                   [0.01494, 4.52194E-6, 5.08E-8],
                   [0.01665, 4.49186E-6, 5.08E-8],
                   [0.01793, 4.46413E-6, 5.08E-8],
                   [0.01935, 4.42796E-6, 5.08E-8],
                   [0.01992, 4.40642E-6, 5.08E-8],
                   [0.02078, 4.39402E-6, 5.08E-8],
                   [0.02149, 4.37807E-6, 5.08E-8],
                   [0.03059, 4.24111E-6, 5.08E-8],
                   [0.04397, 3.96608E-6, 5.08E-8],
                   [0.05891, 3.65661E-6, 5.08E-8],
                   [0.07058, 3.40098E-6, 5.08E-8],
                   [0.08652, 2.99408E-6, 5.08E-8],
                   [0.10004, 2.67464E-6, 5.08E-8],
                   [0.11469, 2.27597E-6, 5.08E-8],
                   [0.12736, 1.75049E-6, 5.08E-8],
                   [0.14372, 1.13134E-6, 5.08E-8],
                   [0.14942, 9.20419E-7, 5.08E-8],
                   [0.15368, 7.10107E-7, 5.08E-8],
                   [0.15653, 5.77418E-7, 5.08E-8],
                   [0.0222, 4.36141E-6, 5.08E-8],
                   [0.05094, 3.81754E-6, 5.08E-8],
                   [0.06546, 3.50055E-6, 5.08E-8],
                   [0.08652, 2.99408E-6, 5.08E-8],
                   [0.10744, 2.41587E-6, 5.08E-8],
                   [0.12736, 1.75049E-6, 5.08E-8],
                   [0.1608, 4.74294E-7, 5.08E-8]]

        nitrogen = [[7.34268E-4, 4.65737E-6, 5.08E-8],
                    [0.00235, 4.52651E-6, 5.08E-8],
                    [0.00351, 4.41434E-6, 5.08E-8],
                    [0.00528, 4.21582E-6, 5.08E-8],
                    [0.00662, 4.07977E-6, 5.08E-8],
                    [0.00803, 3.91843E-6, 5.08E-8],
                    [0.00924, 3.79357E-6, 5.08E-8],
                    [0.01072, 3.5925E-6, 5.08E-8],
                    [0.01214, 3.44254E-6, 5.08E-8],
                    [0.0135, 3.26768E-6, 5.08E-8],
                    [0.01451, 3.12941E-6, 5.08E-8],
                    [0.01551, 2.99225E-6, 5.08E-8],
                    [0.01636, 2.90548E-6, 5.08E-8],
                    [0.01708, 2.81617E-6, 5.08E-8],
                    [0.01765, 2.69253E-6, 5.08E-8],
                    [0.0185, 2.5577E-6, 5.08E-8],
                    [0.01921, 2.46758E-6, 5.08E-8],
                    [0.02006, 2.38549E-6, 5.08E-8],
                    [0.02035, 2.3163E-6, 5.08E-8],
                    [0.0212, 2.17345E-6, 5.08E-8],
                    [0.02206, 2.01902E-6, 5.08E-8],
                    [0.02263, 1.88247E-6, 5.08E-8],
                    [0.02362, 1.7194E-6, 5.08E-8],
                    [0.02433, 1.52829E-6, 5.08E-8],
                    [0.02504, 1.47698E-6, 5.08E-8],
                    [0.02561, 1.31341E-6, 5.08E-8],
                    [0.02633, 1.12088E-6, 5.08E-8],
                    [0.02704, 1.01105E-6, 5.08E-8],
                    [0.02775, 8.39952E-7, 5.08E-8],
                    [0.02832, 7.07161E-7, 5.08E-8]]

        aluminium = [[4.58107E-6, 7E-6, 3.315E-8],
                     [4.96249E-6, 5E-6, 3.315E-8],
                     [4.98934E-6, 5E-6, 3.315E-8],
                     [5.16504E-6, 4E-6, 3.315E-8],
                     [5.52511E-6, 2E-6, 3.315E-8],
                     [5.34736E-6, 3E-6, 3.315E-8],
                     [5.53784E-6, 2E-6, 3.315E-8],
                     [5.86477E-6, 0, 3.315E-8],
                     [4.58001E-6, 7E-6, 3.315E-8],
                     [4.96289E-6, 5E-6, 3.315E-8],
                     [4.98802E-6, 5E-6, 3.315E-8],
                     [4.96063E-6, 4E-6, 3.315E-8],
                     [5.53559E-6, 3E-6, 3.315E-8],
                     [5.86271E-6, 0, 3.315E-8]]

        nickel = [[6.19841E-7, 1E-5, 3.315E-8],
                  [2.09681E-6, 8E-6, 3.315E-8],
                  [2.72334E-6, 7E-6, 3.315E-8],
                  [3.31288E-6, 6E-6, 3.315E-8],
                  [4.65134E-6, 3E-6, 3.315E-8],
                  [3.27078E-6, 6E-6, 3.315E-8],
                  [3.83393E-6, 5E-6, 3.315E-8],
                  [5.48759E-6, 1E-6, 3.315E-8],
                  [5.08614E-6, 2E-6, 3.315E-8],
                  [4.26369E-6, 4E-6, 3.315E-8],
                  [5.8643E-6, 0, 3.315E-8],
                  [6.3383E-7, 1E-5, 3.315E-8],
                  [2.12286E-6, 8E-6, 3.315E-8],
                  [2.76093E-6, 7E-6, 3.315E-8],
                  [1.44766E-6, 9E-6, 3.315E-8],
                  [3.31235E-6, 6E-6, 3.315E-8],
                  [4.67634E-6, 3E-6, 3.315E-8],
                  [3.28318E-6, 6E-6, 3.315E-8],
                  [3.83493E-6, 5E-6, 3.315E-8],
                  [5.48513E-6, 1E-6, 3.315E-8],
                  [5.08965E-6, 2E-6, 3.315E-8],
                  [4.25567E-6, 4E-6, 3.315E-8],
                  [5.85734E-6, 0, 3.315E-8]]

        data = [argon, nitrogen, helium, helium2, nickel, aluminium]

        energy = []
        energy_error = []
        distance = []

        set = self.set
        for counter in range(0, len(data[set])):
            energy.append(data[set][counter][1])
            energy_error.append(data[set][counter][2])
            distance.append(data[set][counter][0])
        # end for

        return energy, energy_error, distance

    def DataFit(self, x, y, error_y, title, xAxisTitle, yAxisTitle, label, degree):
        # y_weights = (1 / error_y) * np.ones(np.size(y))
        y_weights = error_y  # check later #########################################################################
        fit_parameters, fit_errors = np.polyfit(x, y, degree, cov=True, w=y_weights)

        print('Fit np.polyfit of' + title)
        print("The fitting polynomial is of degree " + str(degree))
        if degree == 3:
            print('Constant  term   a = {0} +/- {1}'.format(fit_parameters[3], fit_errors[3][3]))
            print('Linear term      b = {0} +/- {1}'.format(fit_parameters[2], fit_errors[2][2]))
            print('Quadratic term   c = {0} +/- {1}'.format(fit_parameters[1], fit_errors[1][1]))
            print('Cubic term       d = {0} +/- {1}'.format(fit_parameters[0], fit_errors[0][0]))
        elif degree == 1:
            print('Gradient  m = {0} +/- {1}'.format(fit_parameters[1], fit_errors[1][1]))
            print('Intercept d = {0} +/- {1}'.format(fit_parameters[0], fit_errors[0][0]))
        print(" ")

        return fit_parameters, fit_errors  ### be careful with this return it returns an error coviariance matrix not a list

    def returnDifferential(self, b, c, d, x):
        array = []
        function = lambda b, c, d, x: b + 2 * c * x + 3 * d * x ** 2

        # errors for the differential
        errors = []
        function2 = lambda er_d, d, er_b, b, er_c, c, x: np.sqrt(er_b ** 2 + x ** 2 * er_c ** 2 + x ** 4 * er_d ** 2)

        errors = function2(d[1], d[0], b[1], b[0], c[1], c[0], x)
        array = function(b[0], c[0], d[0], x)  # gets the function for each entry of x

        self.differential = -1 * np.array(array)
        self.differential_error = np.array(errors)
        return self.differential, self.differential_error

    def convertChannelNumber(self, channel_number):
        # assumes the channel number is a numpy array
        m = self.coeff[0]
        sigma_m = self.coeff_err[0]
        c = self.coeff[1]
        sigma_c = self.coeff_err[1]

        energy = m * channel_number + c
        energy_error = np.sqrt(channel_number ** 2 * sigma_m ** 2 + sigma_c ** 2)
        return energy, energy_error

    def calibrationCurve(self, type):
        # the type variable determines which type of experiment we are doing
        # the type = True does the calibration curve for gases while the False does the materials. The analysis is
        # then the same for both with one exception of the pressure being converted into effective distance

        fit_param = []
        fit_err = []
        if type:
            # gasses
            V = [42.4, 50, 64, 30.8, 7.92, 6.8, 55.6, 49.6, 42.8, 34.8, 27.4, 13.8, 64, 52.8, 46.8, 38.8, 31, 0.304]
            E = [4.77, 0]
            ch = [439.7, 459.8, 440.45, 220.5, 56.14, 44.45, 399.28, 350.36, 300.7, 249.5, 197.62, 100.07, 458.83,
                  378.18, 333.96, 271.38, 221.4, 0]
            # Volt_err = [3, 3, 1, 1, 1, 0.4, 1, 1, 1, 0.5, 0.5, 0.4, 1, 1, 1, 1, 1, 0.2]
            Volt_err = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                        0.35, 0.35]
            ch_err = np.ones(len(ch)) * 5  # channel number error
            dif_V = 0.5 * (42.4 + 50) - 0.304
            dif_E = 4.77  # MeV

            fit_parameters = [0.25317, 0.10222]  ## obtained from origin
            fit_errors = [[0.15295, "no"], ["no", 8.62475E-4]]

        else:
            # materials
            V = [88.8, 88.8, 104, 10.3, 5.72, 95.2, 85.6, 77.6, 68, 57.6, 47.6, 37, 26.6, 15.9, 5.32, 10.7, 21.4, 31.6,
                 42.4, 52.4, 62.4, 72.4]
            E = [5.8, 0]
            ch = [880.62, 880.52, 880.64, 84.53, 47.86, 806.3, 729.72, 652.29, 570.34, 486.16, 400.41, 312.37, 224.12,
                  134.76, 46.72, 89.56, 178.66, 267.57, 355.56, 442.22, 527.39, 610.54]
            Volt_err = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                        0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
            ch_err = np.ones(len(ch)) * 5
            dif_V = 88.8 - 0.304
            dif_E = 5.8  # MeV
            fit_parameters = [0.23652, 0.10143]  ## obtained from origin
            fit_errors = [[0.12778, "no"], ["no", 2.4636E-4]]

        m_1 = dif_E / dif_V  # two point calibration
        c_1 = m_1 * 0.304 * -1
        m_e = [m_1, 0]
        c_e = [c_1, 0]

        print("Two point claibration fitting variables")
        print("m_e = {0} +- {1}".format(m_e[0], m_e[1]))
        print("c_e = {0} +- {1}".format(c_e[0], c_e[1]))
        # relate voltage to energy
        Vin_Eout = lambda V: m_1 * V + c_1

        # relate channel number to voltage
        # y_weights = Volt_err
        # fit_parameters, fit_errors = np.polyfit(ch, V, 1, cov=True, w=y_weights)

        m_v = [fit_parameters[0], fit_errors[0][0]]
        c_v = [fit_parameters[1], fit_errors[1][1]]

        print("m_v and c_v")
        print(m_v)
        print(c_v)

        def Chin_Vout(ch):
            return m_v[0] * ch + c_v[0]

        function = lambda ch: Vin_Eout(Chin_Vout(ch))

        def energy_error(m_e, m_v, ch, c_v, c_e):
            # each value is an array [value, error]
            # c is a numpy array
            sigma_E = 0
            sigma_E += (m_e[0] * ch[0]) ** 2 * m_v[1] ** 2
            sigma_E += (m_e[0] * m_v[0]) ** 2 * ch[1] ** 2
            sigma_E += m_e[0] ** 2 * c_v[1] ** 2
            sigma_E = np.sqrt(sigma_E)
            return sigma_E

        err_E = []
        for counter in range(0, len(ch)):
            err_E.append(energy_error(m_e, m_v, [ch[counter], ch_err[counter]], c_v, c_e))

        x = np.array(ch)
        y = function(x)

        fit_param, fit_err = np.polyfit(x, y, 1, w=err_E, cov=True)  # final fit with the proper weighting

        # self.coeff = fit_param
        # self.coeff_err = []
        # self.coeff_err.append(fit_err[0][0])
        # self.coeff_err.append(fit_err[1][1])
        print("The calibration curve energy errors")
        print(err_E)
        print("The calibration curve percentage errors")
        print(np.array(err_E) / y * 100)

        mean_err = np.array(err_E).mean()  ## averages the mean error

        ### getting the final calibration values
        k_1 = [0, 0]
        k_2 = [0, 0]
        k_1[0] = m_e[0] * m_v[0]
        k_2[0] = m_e[0] * c_v[0] + c_e[0]
        k_1[1] = np.sqrt((m_e[1] / m_e[0]) ** 2 + (m_v[1] / m_v[0]) ** 2)
        k_2[1] = np.sqrt((m_e[1] / m_e[0]) ** 2 + (c_v[1] / c_v[0]) ** 2)
        k_2[1] = np.sqrt(k_2[1] ** 2 + c_e[1] ** 2)

        self.k_1 = k_1
        self.k_2 = k_2
        plt.plot(x, y, "+")
        plt.xlabel("Channel number")
        plt.ylabel("Energy/ MeV")

        print("Final calibration and errors")
        print("k_1 = {0} +- {1}".format(k_1[0], k_1[1]))
        print("k_2 = {0} +- {1}".format(k_2[0], k_2[1]))

        return mean_err  # returns energy errors
        # assigns the value within the object from the correct calibration

    def getChiSqrt(self, fit_y, y, ey):
        # all arrays are numpy arrays
        # returns the chi squared value
        chi_sqrt = ((y - fit_y) / ey) ** 2
        print("Chi sqr" + str(np.sum(chi_sqrt)))
        return np.sum(chi_sqrt)

    def getChiSqrt2(self, fit_y, y, ey):
        thing = np.abs(y - fit_y)
        print("Thing " + str(np.sum(thing)))
        return np.sum(thing)

    def fitting_I(self, x, y, ey,constant):
        # ey = np.ones(36)
        print("Errors")
        print(ey)
        function = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  / constant # eV m^-1
        # declares the function we want to fit
        limit = 1
        step = 1
        chi_sqr = 10001
        I = 50  # estimate using 10Z eV
        n = 0
        upper = 10000
        stay = True

        while stay:
            if chi_sqr < limit:
                stay = False
                print("limit")
            if n > upper:
                stay = False
                print("iterations")

            if I < 0:
                stay = False
                print("Out of range")

            # check higher
            fit_y = function(4, 2, x, I + step)
            chi_sqr_high = self.getChiSqrt(fit_y, y, ey)
            # check lower
            fit_y = function(4, 2, x, I - step)
            chi_sqr_low = self.getChiSqrt(fit_y, y, ey)
            # adjust I
            if chi_sqr_high < chi_sqr_low:
                I += step
            else:
                I -= step
            n += 1
            print("I = " + str(I))

        self.I = I

        plt.plot(x, y, "b+", label="data")
        print("x and y")
        print(x)
        print(y)
        plt.plot(x, function(4, 2, x, I), "+", label="model")
        plt.legend()
        self.fitting_data = function(4, 2, x, I)
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

    def experimental_cubic(self, set, x):
        # Argon, Nitrogen, Helium, Helium2, Nickel, Aluminium
        all_a = [[-4.16E-04, 4.76E-05], [4.72E-06, 2.79E-08], [4.78E-06, 3.15E-08], [4.77E-06, 2.14E-08],
                 [5.86E-06, 1.22E-08], [5.86E-06, 6.02E-08]]  # intercept
        all_b = [[1.38E-05, 1.15E-05], [2.20E-05, 7.20E-06], [-9.39E-05, 8.53E-06], [-1.08E-04, 6.06E-06],
                 [-0.39226, 0.01115], [-0.11614, 0.07904]]
        all_c = [[-1.87E-05, 7.37E-07],
                 [-0.00494, 4.99E-04],
                 [1.22E-04, 6.18E-04],
                 [8.32E-04, 4.64E-04],
                 [5114.25716, 2646.02899],
                 [-22338.49838, 27789.76664]]
        all_d = [[4.79E-06, 9.80E-09],
                 [0.04246, 0.00986],
                 [-0.0623, 0.01287],
                 [-0.07333, 0.01017],
                 [-1.82E+09, 1.72E+08],
                 [1.83E+09, 2.57E+09]]

        a = all_a[set]
        b = all_b[set]
        c = all_c[set]
        d = all_d[set]

        self.coeff = [a, b, c, d]

        def function(a1, b1, c1, d1, x1):
            return a1[0] + b1[0] * x + c1[0] * x ** 2 + d1[0] * x ** 3

        return function(a, b, c, d, x)

    def analysis(self):
        # loads in the data
        energy, energy_error, distance = self.readData()
        # plots the experimental data
        # plotting the -differential against energy plot with fit

        x = np.array(energy)
        energy = x
        y = self.experimental_cubic(self.set, x)
        y_err = np.array(self.differential_error)

        a = self.coeff[0]
        b = self.coeff[1]
        c = self.coeff[2]
        d = self.coeff[3]
        differential, differential_error = self.returnDifferential(b, c, d, x)

        print(energy)
        print(differential)
        print(differential_error)

        print(len(energy))
        print(len(differential))
        print(len(differential_error))

        differential_error = 0.00001 * differential_error

        # plt.plot(energy, differential)
        plt.errorbar(energy, differential, differential_error)
        plt.xlabel("Energy")
        plt.ylabel("Differential")

        function = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  # eV m^-1
        y = function(4, 2, energy, 10)
        constant = y/differential
        constant = constant.mean()


        self.fitting_I(energy*10, differential, differential_error,constant) ## supposing 2% errors
        print(self.I)
        '''
        figure2 = plt.figure()
        axes_3 = figure2.add_subplot(111)
        #axes_3.plot(energy, differential, "b+", label="Experimental data")
        axes_3.errorbar(energy,differential,"+")
        axes_3.legend()
        axes_3.set_ylabel("-dE/dx")
        axes_3.set_xlabel("E")
        axes_3.set_title("Differential Energy plot")
        I = 20  # first guess
        model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  # eV m^-1
        y_fitted = model(4, 2, energy, I)

        axes_3.plot(x, y_fitted, label="Model")  # fit the model onto the first plot
        axes_4 = figure2.add_subplot(122)
        axes_4.set_xlabel("-dE/dx")
        axes_4.set_ylabel("Error")
        axes_4.errorbar(x, y - y_fitted, yerr=energy_error, fmt='b+')
        plt.savefig("Differential_Plot.png")
        # plots the model on top of it

        # plot the model using the function
        '''


def main():
    # file_name, type = get_choice()
    d = Data(2)
    # load in the datasets
    d.analysis()


main()
plt.show()
