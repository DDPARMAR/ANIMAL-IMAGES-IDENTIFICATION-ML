import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

relative_path = "DATASET"
data = os.listdir(relative_path) 

labels = []
image_array = [] 

for i in data:
    animal_folder = os.listdir("DATASET\\" + i)  # Use double backslashes on Windows
    for j in animal_folder:
        image_path = "DATASET\\" + i + "\\" + j  # Use double backslashes on Windows
        
        # Print a message indicating the image being read
        print("Reading image:", image_path + " Label: " + i)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 600))

        # Flatten the image data and convert to a 1D array
        image = image.flatten()

        labels.append(i)
        image_array.append(image)

# Convert image_array to a numpy array
X = np.array(image_array)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load a new image for prediction
new_image_path = "DATASET\\DOG\\pexels-maria-rosenberg-2171583.jpg"

print("\nPredicting Image: " + new_image_path)
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (600, 600))
new_image = new_image.flatten()
new_image = np.array([new_image])


# Make predictions on the new image
predicted_label = clf.predict(new_image)

# Print the predicted label
print("Predicted Label:", predicted_label[0])
