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

def Argon():
    #### Argon analysis
    # a + bx + cx^2 + dx^3 + ex^4 fx^5
    # fitting values of a 5 degree polynomial onto the stopping power versus distance
    a = -0.00638
    b = 85.73468
    c = 1952.54653
    d = -131683.90214
    e = 7503848.83866
    f = -1.02038E8
    function = lambda b,c,d,e,f,x: b + 2*c*x + 3*d*x**2 + 4*x**3 + 5*f*x**4
    # differential
    model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19) # eV m^-1

    energy_loss = [0.06142, 0.20478, 0.30821, 0.45695, 0.60194, 0.75921, 0.91791, 1.06828, 1.21337, 1.39838, 1.54042, 1.75367, 1.96145, 2.21108, 2.38776, 2.62967, 2.89952, 3.1774, 3.55697, 3.80122, 4.11659, 2.34875, 2.54585, 2.80584, 3.10851, 3.40752, 3.68468, 3.98613, 4.41427]
    distance = [0.000700116, 0.00238, 0.0036, 0.00501, 0.00638, 0.00795, 0.00928, 0.01077, 0.01202, 0.01363, 0.0148, 0.01636, 0.01793, 0.01921, 0.02078, 0.0222, 0.02334, 0.02504, 0.02647, 0.02789, 0.02917, 0.02006, 0.0212, 0.02263, 0.02419, 0.02547, 0.02704, 0.02832, 0.03003]

    energy_loss = np.array(energy_loss)
    distance = np.array(distance)

    model_values = model(4,2,energy_loss, 15)
    constant = model_values/function(b,c,d,e,f,energy_loss)
    plt.plot(energy_loss, model_values, label="model")
    plt.plot(energy_loss, function(b,c,d,e,f,energy_loss)*constant,"+", label="differential")

    plt.legend()
    plt.show()
    ### errors for argon



def Helium1():
    a = [-0.01108,	0.04096]
    b = [28.96517,	24.01517]
    c = [-5433.52023,	4396.24059]
    d = [776569.25218,	332637.91901]
    e = [-2.7429E7,	1.09724E7]
    f = [3.27238E8,	1.30651E8]
    energy_loss = [0.0533, 0.06427, 0.08327, 0.10725, 0.17004, 0.21372, 0.30852, 0.41835, 0.57156, 0.68454, 0.84161, 1.01321, 1.18492, 1.39045, 1.53473, 1.76028, 1.90069, 2.10064, 2.40635, 2.619, 3.1522]
    distance = [-0.0025, -0.00388, -0.00499, -0.00632, -0.00795, -0.00928, -0.01073, -0.01221, -0.0138, -0.01508, -0.01651, -0.01779, -0.01921, -0.02078, -0.0222, -0.02391, -0.02504, -0.02647, -0.02775, -0.02974, -0.03273]
    energy_loss = np.array(energy_loss)
    distance = np.array(distance)
    function = lambda b, c, d, e, f, x: b + 2 * c * x + 3 * d * x ** 2 + 4 * x ** 3 + 5 * f * x ** 4
    # differential
    model = lambda N, Z, E, I: -3.801 * (N * Z / E) * (np.log(E) + 6.307 - np.log(I)) * 10 ** (-19)  # eV m^-1

    model_values = model(4, 2, energy_loss, 15)
    constant = model_values / function(b, c, d, e, f, energy_loss)
    plt.plot(energy_loss, model_values, label="model")
    plt.plot(energy_loss, function(b, c, d, e, f, energy_loss) * constant, "+", label="differential")

    plt.legend()
    plt.show()

Helium1()
