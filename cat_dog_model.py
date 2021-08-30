# Dataset - archive

print("======================================")
print()
print("Cat and dog Classifier")
print()
print("======================================")


# Import the libraries

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg

# Set the path

training_path = 'C:/Users/Anand/Desktop/My_Files/Learnbay_Docs/DeepLearning/CatorDog_classifier/archive/training_set/training_set'
testing_path ='C:/Users/Anand/Desktop/My_Files/Learnbay_Docs/DeepLearning/CatorDog_classifier/archive/test_set/test_set'

# Image generator function 

train_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1/255)

# Connecting training and testing data

training_set = train_datagen.flow_from_directory(training_path, target_size=(64,64), class_mode='binary', batch_size=32)
testing_set = test_datagen.flow_from_directory(testing_path, target_size=(64,64), class_mode='binary', batch_size=32)


# Visualizing the data

training_data = next(training_set)
print("======================================")
print()
print("Training data ---->  ", training_data)
print()
print("======================================")

# Shape of training data
print("======================================")
print()
print("Shape of the training data ----> ",training_data[0].shape)
print()
print("======================================")

# Image of example data 

print("======================================")
print()
print(" Image 1 ")
plt1 = plt.imshow(training_data[0][0])
plt.show()
print()
print("======================================")


print("======================================")
print()
print(" Image 2 ")
plt2 = plt.imshow(training_data[0][1])
plt.show()
print()
print("======================================")

print("======================================")
print()
print(" Image 3 ")
plt3 = plt.imshow(training_data[0][10])
plt.show()
print()
print("======================================")


 #Steps :
 #1. Build a model architecture
 #2. Compilation
 #3. Fitting the model

#Step 1 : Building the model architechture
print()
print("======================================")
print("Building the model....")
print()
print("======================================")

classifier = Sequential()

# CNN LAYER
classifier.add(Conv2D(32, 3, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D())


# Flattening 
classifier.add(Flatten())

# ANN LAYER
# Hidden Layer 1
classifier.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.2))

# Output Layer
classifier.add(Dense(1, activation='sigmoid'))

 #Step 2 : Compilation
print()
print("======================================")
print("Compiling the model....")
print()
print("======================================")


classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])



#Step 3 : Fitting the model
print()
print("======================================")
print("Fitting the model....")
classifier.fit(training_set, batch_size = 2048, epochs=20, validation_data=testing_set )
print()
print("Model Fitting completed....")
print("======================================")


# Saving the model
classifier.save('Cat_Dog_Classifier.h5')

#Read the image, resize the image and predict the image
test_image= image.load_img('C:/Users/Anand/Desktop/My_Files/Learnbay_Docs/DeepLearning/CatorDog_classifier/archive/single prediction/cat_or_dog_2.jpg', target_size =(64,64))

print()
print("======================================")
print("Prediction Image")
print()
print("======================================")

img1 = mpimg.imread('C:/Users/Anand/Desktop/My_Files/Learnbay_Docs/DeepLearning/CatorDog_classifier/archive/single prediction/cat_or_dog_2.jpg')
imgplot = plt.imshow(img1)
plt.show()

test_image =image.img_to_array(test_image)
test_image =test_image.reshape(1,64,64,3)
result = classifier.predict(test_image)

print()
print("======================================")
print("Predictions of result---->",result[0])
print()
print("======================================")

print()
print("======================================")
print("Classification rule : ",training_set.class_indices)
print()
print("======================================")



if [result[0]>=0.5][0] == True:
  print()
  print("======================================")
  print("Prediction result : DOG")
  print()
  print("======================================")
  
elif [result[0]<0.5][0]== True:
  print()
  print("======================================")
  print("Prediction result : CAT")
  print()
  print("======================================")

