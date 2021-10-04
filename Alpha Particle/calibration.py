import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self, fileName):
        self.x = []
        self.y = []
        self.error = []
        self.ErrorCheck = True
        self.fileName = fileName[:len(fileName)-1]

        if self.fileName[len(fileName) - 4:] == "txt":
            self.readData()



        self.x_div = np.abs(self.x[0] - self.x[1])
        print("The x division is: " + str(self.x_div))
        self.signals = []

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
        temp2 = 0
        file = open(self.fileName)
        content = file.readlines()
        for counter in range(0, len(content)):
            temp = content[counter]
            if temp == "end\n":
                temp2 += 1
            else:
                array[temp2].append(float(temp))
            # end if
        # end for
        self.x = array[0]
        self.y = array[1]
        self.error = array[2]
        if len(self.error) == 0:
            self.ErrorCheck = False

        file.close()

    # organize data
    # show the plot

    def loadCSVFile(self):
        x_array = []
        y_array = []
        file = open(self.fileName)
        content = file.readlines()
        for counter in range(0, len(content), 1):
            entry = content[counter]
            temp = entry.split(",")

            # heading entry
            x = float(temp[3])
            y = float(temp[4])
            self.x.append(x)
            self.y.append(y)

def getfile():
    file = open("file_names.txt","r")
    content = file.readlines()
    return content



def main():

    file = getfile()
    print(file)
    choice = int(input("Enter number: "))
    d = Data(path + "\\Data\\" + file[choice])
    # C:\Users\opopn\Desktop\Labs code\Data
    d.plotCustom()
    print(file[choice])



main()
plt.show()