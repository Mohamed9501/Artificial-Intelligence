import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data_frame = pd.read_csv('breast_cancer_svm/data.csv', delimiter=',', nrows=None)
data_frame.dataframeName = 'data.csv'

# Drop the labels from the input data
X = data_frame.drop(['diagnosis'], axis=1)
# Drop the ID column which is not needed.
X = X.drop(['id'], axis=1)
# Create the labels variable
Y = data_frame['diagnosis']
# Drop NA data
X = X.dropna(axis=1)
# Normalization
X_scaled = X / (X.max())
# Split data into training and testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=40)
# Initialize SVM model, and fit the data.
svc_model = SVC()
svc_model.fit(X_train, Y_train)
# Predict the output.
y_predict = svc_model.predict(X_test)
# Print the classification results.
print(classification_report(Y_test, y_predict))
