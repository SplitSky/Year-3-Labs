import numpy as np
import matplotlib.pyplot as plt

### expected ionisation energy values
'''
Aluminium	13  	5.98577
Nickel	    28  	7.6398
Helium	    2   	24.58738
Argon	    18  	15.75962
Nitrogen	7   	14.53414
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:19:14 2021

@author: arencatonyener
"""

# Simpson's 1/3 Rule

# Define function to integrate

import numpy as np

N = 4
Z = 2
I = 5


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


def fitting_I(x, y):
    # ey = np.ones(36)
    x = x
    function = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  # eV m^-1
    # declares the function we want to fit
    limit = 1
    step = 0.01
    chi_sqr = 10001
    I = 15  # estimate using 10Z eV
    n = 0
    upper = 100
    stay = True

    while stay:
        if chi_sqr < limit:
            stay = False
            print("limit")
        if n > upper:
            stay = False
            print("iterations")

        if I <= 0:
            stay = False
            print("Out of range")

        # check higher
        fit_y = function(4, 2, x, I + step)
        chi_sqr_high = getChiSqrt(fit_y, y)
        # check lower
        fit_y = function(4, 2, x, I - step)
        chi_sqr_low = getChiSqrt(fit_y, y)
        # adjust I
        print("Chi low: " + str(chi_sqr_low))
        print("Chi high: " + str(chi_sqr_high))
        if chi_sqr_high < chi_sqr_low:
            I += step
            chi_sqr = chi_sqr_high
        elif chi_sqr_high > chi_sqr_low:
            I -= step
            chi_sqr = chi_sqr_low
        else:
            I = I
            print("Oscillating")
            stay = False
            # break out of the loop because the value sees no improvement

        n += 1
        print("I = " + str(I))

    # plt.plot(x, y, "b+", label="data")
    print("x and y")
    # plt.plot(x, function(4, 2, x, I), "+", label="model")
    plt.legend()
    fitting_data = function(4, 2, x, I)
    return I


def Argon():
    #### Argon analysis
    # a + bx + cx^2 + dx^3 + ex^4 fx^5
    # fitting values of a 5 degree polynomial onto the stopping power versus distance

    a = [4.77638, 0.05115]
    b = [-85.73468, 32.34848]
    c = [-1952.54653, 6315.76888]
    d = [131683.90214, 508997.75393]
    e = [-7503848.83866, 1.79768E7]
    f = [1.02038E8, 2.30634E8]

    def function2(x):
        a = [4.77638, 0.05115]
        b = [-85.73468, 32.34848]
        c = [-1952.54653, 6315.76888]
        d = [131683.90214, 508997.75393]
        e = [-7503848.83866, 1.79768E7]
        f = [1.02038E8, 2.30634E8]

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

    energy = 4.77 - np.array(energy_loss)

    print("Energy: ")
    print(energy)

    distance = np.array(distance)

    model_values = model(4, 2, energy_loss, 15)
    plt.plot(energy_loss, -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss), "m+", label="Argon")
    differential = function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    print("Differential")
    print(differential)

    plt.errorbar(energy_loss, differential, differential * 0.10, fmt="m+")


def Helium1():
    a = [-0.01108, 0.04096]
    b = [28.96517, 24.01517]
    c = [-5433.52023, 4396.24059]
    d = [776569.25218, 332637.91901]
    e = [-2.7429E7, 1.09724E7]
    f = [3.27238E8, 1.30651E8]
    energy_loss = [0.0533, 0.06427, 0.08327, 0.10725, 0.17004, 0.21372, 0.30852, 0.41835, 0.57156, 0.68454, 0.84161,
                   1.01321, 1.18492, 1.39045, 1.53473, 1.76028, 1.90069, 2.10064, 2.40635, 2.619, 3.1522]
    distance = [-0.0025, -0.00388, -0.00499, -0.00632, -0.00795, -0.00928, -0.01073, -0.01221, -0.0138, -0.01508,
                -0.01651, -0.01779, -0.01921, -0.02078, -0.0222, -0.02391, -0.02504, -0.02647, -0.02775, -0.02974,
                -0.03273]

    energy_loss = np.array(energy_loss)
    distance = -1 * np.array(distance)

    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  # eV m^-1

    model_values = model(4, 2, energy_loss, 2)
    # constant = model_values / function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    # constant = np.mean(constant)
    # plt.plot(energy_loss, model_values, "+", label="model")

    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    plt.plot(energy_loss, -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss), "+", label="Helium 1")


def Helium2():
    a = [-0.01763, 0.01314]
    b = [16.97025, 1.88807]
    c = [106.52984, 81.56778]
    d = [-2155.39018, 1397.72986]
    e = [20993.39666, 10102.44543]
    f = [-57770.64359, 25820.20065]
    energy_loss = [-0.00512, 0.01073, 0.05502, 0.06813, 0.09902, 0.11924, 0.16048, 0.1808, 0.19798, 0.2251, 0.24806,
                   0.27814, 0.30587, 0.34204, 0.36358, 0.37598, 0.39193, 0.52889, 0.80392, 1.11339, 1.36902, 1.77592,
                   2.09536, 2.49403, 3.01951, 3.63866, 3.84958, 4.05989, 4.19258, 0.40859, 0.95246, 1.26945, 1.77592,
                   2.35413, 3.01951, 4.29571]
    distance = [0.0008538, 0.00215, 0.00406, 0.00515, 0.00663, 0.0078, 0.00936, 0.01083, 0.01214, 0.01365, 0.01494,
                0.01665, 0.01793, 0.01935, 0.01992, 0.02078, 0.02149, 0.03059, 0.04397, 0.05891, 0.07058, 0.08652,
                0.10004, 0.11469, 0.12736, 0.14372, 0.14942, 0.15368, 0.15653, 0.0222, 0.05094, 0.06546, 0.08652,
                0.10744, 0.12736, 0.1608]

    energy_loss = np.array(energy_loss)
    distance = np.array(distance)
    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)

    ey = get_diff_error(b, c, d, e, f, energy_loss)
    plt.plot(energy_loss, differential, "b+", label="Helium 2")
    plt.errorbar(energy_loss, differential, differential * 0.10, fmt="b+")


def Nitrogen():
    a = [0.06525, 0.03177]
    b = [59.3309, 21.39043]
    c = [8793.90858, 4395.99209]
    d = [-677621.08937, 372224.88171]
    e = [2.53573E7, 1.38091E7]
    f = [-3.10349E8, 1.86203E8]
    energy_loss = [0.11263, 0.24349, 0.35566, 0.55418, 0.69023, 0.85157, 0.97643, 1.1775, 1.32746, 1.50232, 1.64059,
                   1.77775, 1.86452, 1.95383, 2.07747, 2.2123, 2.30242, 2.38451, 2.4537, 2.59655, 2.75098, 2.88753,
                   3.0506, 3.24171, 3.29302, 3.45659, 3.64912, 3.75895, 3.93005, 4.06284]

    distance = [-0.000734268, -0.00235, -0.00351, -0.00528, -0.00662, -0.00803, -0.00924, -0.01072, -0.01214, -0.0135,
                -0.01451, -0.01551, -0.01636, -0.01708, -0.01765, -0.0185, -0.01921, -0.02006, -0.02035, -0.0212,
                -0.02206, -0.02263, -0.02362, -0.02433, -0.02504, -0.02561, -0.02633, -0.02704, -0.02775, -0.02832]

    energy_loss = np.array(energy_loss)
    distance = -1 * np.array(distance)
    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    plt.plot(energy_loss, differential, "g+", label="Nitrogen")
    plt.errorbar(energy_loss, differential, differential * 0.10, fmt="g+")


def Aluminium():
    a = [-0.06297, 0.06021]
    b = [116143.67261, 79036.3635]
    c = [2.23385E10, 2.77898E10]
    d = [-1.83487E15, 2.57161E15]
    e = [-2.60596E10, 3.6523E10]
    f = [-257487.71374, 360873.75439]
    energy_loss = [1.21893, 0.83751, 0.81066, 0.63496, 0.27489, 0.45264, 0.26216, -0.06477, 1.21999, 0.83711, 0.81198,
                   0.83937, 0.26441, -0.06271]

    distance = [-7e-06, -5e-06, -5e-06, -4e-06, -2e-06, -3e-06, -2e-06, 0, -7e-06, -5e-06, -5e-06, -4e-06, -3e-06, 0]

    energy_loss = np.array(energy_loss)
    distance = -1 * np.array(distance)
    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    plt.plot(energy_loss, differential, "k+", label="Aluminium")
    plt.errorbar(energy_loss, differential, differential * 0.30, fmt="k+")


def Nickel():
    a = [-0.06436, 0.01268]
    b = [391097.88528, 11903.6986]
    c = [-4.73144E9, 2.92109E9]
    d = [1.7973E15, 1.96189E14]
    e = [3.50385E10, 3.82471E9]
    f = [482402.88167, 52657.82006]
    energy_loss = [5.18016, 3.70319, 3.07666, 2.48712, 1.14866, 2.52922, 1.96607, 0.31241, 0.71386, 1.53631, -0.0643,
                   3.67714, 4.35234, 2.48765, 1.12366, 2.51682, 1.96507, 0.31487, 0.71035, 1.54433, -0.05734]

    distance = [1e-05, 8e-06, 7e-06, 6e-06, 3e-06, 6e-06, 5e-06, 1e-06, 2e-06, 4e-06, 0, 8e-06, 9e-06, 6e-06, 3e-06,
                6e-06, 5e-06, 1e-06, 2e-06, 4e-06, 0]

    energy_loss = np.array(energy_loss)
    distance = np.array(distance)
    function = lambda b1, c1, d1, e1, f1, x1: b1 + 2 * c1 * x1 + 3 * d1 * x1 ** 2 + 4 * x1 ** 3 + 5 * f1 * x1 ** 4
    # differential
    differential = -1 * function(b[0], c[0], d[0], e[0], f[0], energy_loss)
    plt.plot(energy_loss, differential, "c+", label="Nickel")
    plt.errorbar(energy_loss, differential, differential * 0.10, fmt="c+")


Argon()
# Helium1()
# Helium2()
# Nitrogen()
# Nickel()
# Aluminium()

plt.legend()

plt.xlabel("Energy loss / MeV")
plt.ylabel("-dE/dx / eVm^-1")
plt.title("Stopping power against energy loss")
plt.show()


