import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

relative_path = "DATASET"
data = os.listdir(relative_path)

labels = []
image_data = []

for i in data:
    animal_folder = os.listdir(os.path.join(relative_path, i))
    
    for j in animal_folder:
        image_path = os.path.join(relative_path, i, j)

        print("Reading image:", image_path + " Label: " + i)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 600)) 

        image = image.flatten()
        labels.append(i)
        image_data.append(image)


X = np.array(image_data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_image = cv2.imread("TEST\\testlion1.jpg") 
new_image = cv2.resize(new_image, (600, 600)) 

new_image_data = new_image.flatten()

predicted_label = clf.predict([new_image_data])  
print("Predicted Label:", predicted_label[0])