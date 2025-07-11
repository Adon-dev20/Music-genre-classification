import json
import numpy as np
import keras
from keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

DATA_PATH = "DataMFCC.json"

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfccs"])
    y = np.array(data["labels"])
    labels = data["genres"]
    return X, y, labels

def prepare_datasets(test_size, validation_size):
    X, y, labels = load_data(DATA_PATH)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp,
                                                                  test_size=(test_size / (test_size + validation_size)))
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test, labels

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = cm * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.xticks(rotation=45)
    plt.show()

def build_model(input_shape):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test, labels = prepare_datasets(0.15, 0.15)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100,
                        callbacks=[early_stopping, reduce_lr])

    plot_history(history)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, labels)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))




