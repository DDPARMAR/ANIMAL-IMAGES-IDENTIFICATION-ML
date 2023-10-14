import cv2

image_path = "D:\ANIMAL-IMAGES-IDENTIFICATION-ML\DATASET\CAT\cat-10.jpg"
image = cv2.imread(image_path)

if image is not None:
    print("Image loaded successfully.")
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image.")
