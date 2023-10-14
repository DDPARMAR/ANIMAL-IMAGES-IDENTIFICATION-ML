import os

relative_path = "DATASET"
data = os.listdir(relative_path)

for i in data:
	animal_folder = os.listdir("DATASET/" + i)
	print(animal_folder)
	for j in animal_folder:
		print(j)