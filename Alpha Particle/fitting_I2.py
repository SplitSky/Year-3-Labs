import matplotlib.pyplot as plt

### expected ionisation energy values
'''
Aluminium	13  	5.98577
Nickel	    28  	7.6398
Helium	    2   	24.58738
Argon	    18  	15.75962
Nitrogen	7   	14.53414

// only value for Helium is correct

'''
density_N = [6.04e+28, 9.24e+28, 5e+25, 2.5e+25, 2.5e+25]
# order of density numbers is:
# Aluminium-27
# Nickel-58
# Nitrogen-14
# Argon-40
# Helium-4
### use Helium to verify the data works


# Simpson's 1/3 Rule

# Define function to integrate

import numpy as np

Z = 2


def Simpson(lower_limit, upper_limit, sub_interval, function):
    f = function

    # Implementing Simpson's 1/3
    def simpson13(x0, xn, n):
        # calculating step size
        h = (xn - x0) / n

        # Finding sum
        integration = f(x0) + f(xn)

        for i in range(1, n):
            k = x0 + i * h

            if i % 2 == 0:
                integration = integration + 2 * f(k)
            else:
                integration = integration + 4 * f(k)

        # Finding final integration value
        integration = integration * h / 3

        return integration

    # Input section
    # Call trapezoidal() method and get result
    result = simpson13(lower_limit, upper_limit, sub_interval)
    print("Integration result by Simpson's 1/3 method is: {0}".format(result))
    return result


def get_diff_error(b, c, d, e, f, x):
    temp = (b[1] ** 2)
    temp1 = ((2 * x) ** 2 * c[1] ** 2)
    temp2 = ((3 * x ** 2) ** 2 * d[1] ** 2)
    temp3 = ((4 * x ** 3) ** 2 * e[1] ** 2)
    temp4 = (5 * x ** 4) ** 2 * f[1] ** 2
    return temp + temp1 + temp2 + temp3 + temp4  # returns numpy array of errors


def getChiSqrt(fit_y, y):
    # all arrays are numpy arrays
    # returns the chi squared value
    chi_sqrt = (y - fit_y) ** 2
    return np.sum(chi_sqrt)


def fitting_I(x, y, I_init, N):
    # ey = np.ones(36)
    I = I_init
    x = x
    function = lambda E: -3.801 * (N * 2 / E) * (np.log(E) + 6.307 - np.log(I)) * 1E-19  # eV m^-1

    # plt.plot(x, y, "b+", label="data")
    # print("x and y")
    # plt.plot(x, function(4, 2, x, I), "+", label="model")
    # plt.legend()
    fitting_data = function(x)

    return I


def Argon():
    #### Argon analysis
    # a + bx + cx^2 + dx^3 + ex^4 fx^5
    # fitting values of a 5 degree polynomial onto the stopping power versus distance

    a = [-0.00638, 0.05115]
    b = [85.73468, 32.34848]
    c = [1952.54653, 6315.76888]
    d = [-131683.9021, 508997.7539]
    e = [7503848.839, 1.80E+07]
    f = [-1.02E+08, 2.31E+08]

    def function2(x):
        a = [-0.00638, 0.05115]
        b = [85.73468, 32.34848]
        c = [1952.54653, 6315.76888]
        d = [-131683.9021, 508997.7539]
        e = [7503848.839, 1.80E+07]
        f = [-1.02E+08, 2.31E+08]
        return 1 / (b[0] + 2 * c[0] * x + 3 * d[0] * x ** 2 + 4 * e[0] * x ** 3 + 5 * f[0] * x ** 4)

    def function3(x):
        model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-13)  # MeV m^-1
        Z = 18
        I = 15
        N = 0.0025 * 10 ** 28
        y = 1 / model(N, Z, x, I)
        return y

    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-13)  # MeV m^-1

    energy_loss = [0.06142, 0.20478, 0.30821, 0.45695, 0.60194, 0.75921, 0.91791, 1.06828, 1.21337, 1.39838, 1.54042,
                   1.75367, 1.96145, 2.21108, 2.38776, 2.62967, 2.89952, 3.1774, 3.55697, 3.80122, 4.11659, 2.34875,
                   2.54585, 2.80584, 3.10851, 3.40752, 3.68468, 3.98613, 4.41427]
    distance = [0.000700116, 0.00238, 0.0036, 0.00501, 0.00638, 0.00795, 0.00928, 0.01077, 0.01202, 0.01363, 0.0148,
                0.01636, 0.01793, 0.01921, 0.02078, 0.0222, 0.02334, 0.02504, 0.02647, 0.02789, 0.02917, 0.02006,
                0.0212, 0.02263, 0.02419, 0.02547, 0.02704, 0.02832, 0.03003]

    length = len(energy_loss)

    energy_loss = np.array(energy_loss)
    energy = 4.77 - energy_loss
    print("zdskjudawhknj")
    print(energy)
    distance = np.array(distance)

    model_values = model(4, 2, energy_loss, 15)
    # constant = model_values / function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    # plt.plot(energy_loss, model_values, "+", label="model")
    # plt.plot(energy_loss, function(b[0], c[0], d[0], e[0], f[0], energy_loss), "+", label="data")

    I = 15

    # plt.xlabel("Energy loss / MeV")
    # plt.ylabel("-dE/dx / eVm^-1")
    # plt.title("Argon")

    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    range1 = []

    for counter in range(0, length, 1):
        range1.append(Simpson(energy[counter], 4.77, 500, function2))

    print("Values of range")
    print(range1)

    plt.plot(np.array(range1) * -1, energy, "+", label="experimental")
    plt.title("Effective Range of alpha particle- Argon")
    plt.xlabel("Average Range/ m")
    plt.ylabel("Particle Energy/ MeV")

    N = 0.0025 * 10 ** 28
    range1 = []
    for counter in range(0, length, 1):
        range1.append(Simpson(energy[counter], 4.77, 500, function3))

    print("Theoretical values")

    print(range1)
    plt.plot(np.array(range1) * -1, energy_loss, "+", label="theoretical")
    plt.legend()
    plt.show()


def Helium2():
    a = [4.78763, 0.01314]
    b = [-16.97025, 1.88807]
    c = [-106.52984, 81.56778]
    d = [2155.39018, 1397.72986]
    e = [-20993.39666, 10102.44543]
    f = [57770.64359, 25820.20065]

    N = density_N[4]
    I = 40

    def function2(x):
        a = [4.78763, 0.01314]
        b = [-16.97025, 1.88807]
        c = [-106.52984, 81.56778]
        d = [2155.39018, 1397.72986]
        e = [-20993.39666, 10102.44543]
        f = [57770.64359, 25820.20065]

        return 1 / (b[0] + 2 * c[0] * x + 3 * d[0] * x ** 2 + 4 * e[0] * x ** 3 + 5 * f[0] * x ** 4)

    def function3(x):
        # the theoretical function for obtaining the range
        return 3.801 * (N * 2 / x) * (np.log(np.abs(x)) + 6.307 - np.log(I)) * 1E-19  # eV m^-1

    energy_loss = [-0.00512, 0.01073, 0.05502, 0.06813, 0.09902, 0.11924, 0.16048, 0.1808, 0.19798, 0.2251, 0.24806,
                   0.27814, 0.30587, 0.34204, 0.36358, 0.37598, 0.39193, 0.52889, 0.80392, 1.11339, 1.36902, 1.77592,
                   2.09536, 2.49403, 3.01951, 3.63866, 3.84958, 4.05989, 4.19258, 0.40859, 0.95246, 1.26945, 1.77592,
                   2.35413, 3.01951, 4.29571]  # energy loss in MeV
    distance = [0.0008538, 0.00215, 0.00406, 0.00515, 0.00663, 0.0078, 0.00936, 0.01083, 0.01214, 0.01365, 0.01494,
                0.01665, 0.01793, 0.01935, 0.01992, 0.02078, 0.02149, 0.03059, 0.04397, 0.05891, 0.07058, 0.08652,
                0.10004, 0.11469, 0.12736, 0.14372, 0.14942, 0.15368, 0.15653, 0.0222, 0.05094, 0.06546, 0.08652,
                0.10744, 0.12736, 0.1608]  # in meters

    energy_loss = np.array(energy_loss)
    distance = np.array(distance)
    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)

    ey = get_diff_error(b, c, d, e, f, energy_loss)

    energy = energy_loss
    ranges = []
    temp = 0
    for entry in energy_loss:
        temp = Simpson(entry, 4.77, 10000, function2)
        ranges.append(temp)
    ranges = np.array(ranges)
    print("Ranges:")
    print(ranges)

    plt.plot(ranges, energy_loss, "b+", label="range experimental")
    plt.xlabel("Average Range/ m")
    plt.ylabel("Particle Energy/ MeV")
    ey = energy_loss * 0.058
    plt.errorbar(ranges, energy_loss, ey, fmt="b+")

    # calculate the theoretical values
    ranges2 = []
    for entry in energy_loss:
        temp = Simpson(entry, 4.77, 10000, function3)
        ranges2.append(temp)

    ranges2 = np.array(ranges2)
    print("Ranges 2")
    print(ranges2)
    plt.plot(ranges2, energy_loss, "c+", label="range theoretical")
    ey2 = energy_loss * 0.058
    plt.errorbar(ranges2, energy_loss, ey, fmt="c+")


Helium2()
plt.show()


