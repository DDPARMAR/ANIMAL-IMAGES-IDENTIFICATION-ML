import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

relative_path = "DATASET"
data = os.listdir(relative_path)

labels = []
feature_vectors = []

for i in data:
    animal_folder = os.listdir(os.path.join(relative_path, i))
    for j in animal_folder:
        image_path = os.path.join(relative_path, i, j)

        print("Reading image:", image_path + " Label: " + i)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (500, 500))  # Resize to a common size

        image = image.flatten()
        labels.append(i)
        feature_vectors.append(image)

# Convert feature_vectors to a numpy array
X = np.array(feature_vectors)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier with fine-tuned parameters
clf = svm.SVC(kernel='linear', C=1, gamma='scale')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# To make predictions on a new image:
new_image = cv2.imread("TEST\\testlion1.jpg")  # Use the correct path separator
new_image = cv2.resize(new_image, (500, 500))  # Resize to match the training data dimensions

new_features = new_image.flatten()

predicted_label = clf.predict([new_features])  # Use the trained classifier
print("Predicted Label:", predicted_label[0])
