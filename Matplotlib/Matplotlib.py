import matplotlib.pyplot as plt
import pandas as pd

X1 = [5, 1, 1, 1, 5, 1, 1, 1]
Y1 = [12, 12, 8, 6, 6, 6, 4, 2]

X2 = [6, 6, 6, 8, 10, 12, 12, 12]
Y2 = [12, 8, 6, 4, 4, 6, 8, 12]

X3 = [17, 16, 15, 14, 13, 13, 14, 15, 16, 17]
Y3 = [9, 11, 11, 9, 7, 5, 3, 1, 1, 3]


# data = pd.read_csv('E:\\0_code\\Python\\Machine Learning\\RockVSMine\\Sonar.csv', header = None) #Read the data from the csv file

plt.plot(X1, Y1)
plt.plot(X2, Y2)
plt.plot(X3, Y3)
plt.show()