import numpy as np  #To perform mathematical operations
import pandas as pd #To read the data from the csv file

from sklearn.model_selection import train_test_split  #To split the data
from sklearn.linear_model import LogisticRegression   #To create the model
from sklearn.metrics import accuracy_score  #To calculate the accuracy of the model

# import matplotlib.pyplot as plt #To plot the data

# i need to plot the data

sonar_data = pd.read_csv('E:\\0_code\\Python\\Machine Learning\\RockVSMine\\Sonar.csv', header = None) #Read the data from the csv file


print(sonar_data.shape)
# print(plt.plot(sonar_data.shape))
# plt.show()

# Assuming the first two columns are numeric
# plt.scatter(sonar_data[0], sonar_data[1])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter plot of Feature 1 vs Feature 2')
# plt.show()

print(sonar_data.describe()) #Statistics of the data
print(sonar_data[60].value_counts()) #Count of the number of rocks and mines

sonar_data.groupby(60).mean()   #Mean of the data

X = sonar_data.drop(columns = 60, axis = 1) #separating the data into features
Y = sonar_data[60] #separating the data into labels

# plt.plot(X, Y)
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1) #Splitting the data into training and testing data

model = LogisticRegression() #Creating the model

model.fit(X_train, Y_train) #Training the model

X_train_prediction = model.predict(X_train) #Testing the model
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) #Calculating the accuracy of the model

print('Accuracy on training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test) #Testing the model
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) #Calculating the accuracy of the model

print('Accuracy on test data : ', test_data_accuracy)

print('Test data accuracy:', test_data_accuracy)

# plt.plot(X_train, X_train_prediction)
# plt.text(0.5, 0.5, 'Test data accuracy: {:.2f}'.format(test_data_accuracy),
#          horizontalalignment='center', verticalalignment='center',
#          transform=plt.gca().transAxes)
# plt.show()


predicting_sample = [float(i) for i in input('sample to predict').split(',')]
predicting_sample_reshaped = np.asarray(predicting_sample).reshape(1, -1)
prediction = model.predict(predicting_sample_reshaped)

print(prediction)