import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import os
from skimage import io, color, feature
from sklearn.model_selection import train_test_split 
import cv2 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import KFold  
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

def extract_hog_feature(img):
    gray_img = color.rgb2gray(img)
    hog_feature , hog_img = feature.hog(gray_img, visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return hog_feature, hog_img_rescaled

def classification_report_nn(y_true, y_pred, classes):
    return classification_report(y_true, y_pred, target_names=classes)

def plot_roc_curve_multi_class(y_true, y_score, classes):
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for class {classes[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

dataset_path = 'test4'

class_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
features_list = []
labels_list = []

for class_folder in class_folders:
    class_name = os.path.basename(class_folder)

    # Loop through each image in the class folder
    for image_filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_filename)

        # Load the image
        image = io.imread(image_path)

        # Resize the image to 64x64
        resized_image = cv2.resize(image, (64, 64))

        # Extract HOG features and visualize
        hog_features, hog_image = extract_hog_feature(resized_image)

        # # Display the original image and the HOG features
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 2), sharex=True, sharey=True)

        # ax1.axis('off')
        # ax1.imshow(resized_image, cmap=plt.cm.gray)

        # ax2.axis('off')
        # ax2.imshow(hog_image, cmap=plt.cm.gray)

        # plt.show()

        # Append HOG features to the features list
        features_list.append(hog_features)

        # Append the label to the labels list
        labels_list.append(class_name)

features_array = np.array(features_list)
labels_array = np.array(labels_list)
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels_array)
features_array, numeric_labels = shuffle(features_array, numeric_labels, random_state=74)

# Split the data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(features_array, numeric_labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),      
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,  # Stop training if no improvement after 3 epochs
    restore_best_weights=True  # Restore weights of the best model
)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,  # Maximum number of epochs
    batch_size=64,
    callbacks=[early_stopping_callback]  # Add early stopping callback
)




# Evaluate the model
y_pred_nn = model.predict(X_test)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

# Print the classification report
classification_rep_nn = classification_report_nn(y_test, y_pred_nn_classes, label_encoder.classes_)
print("Classification Report for Neural Network Model:")
print(classification_rep_nn)



# Plot the ROC curve
plot_roc_curve_multi_class(y_test, y_pred_nn, label_encoder.classes_)


# Plot the confusion matrix
plot_confusion_matrix(y_test, y_pred_nn_classes, label_encoder.classes_)