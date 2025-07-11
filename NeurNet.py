import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras as keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

dataset = "DataMFCC.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)


    X = np.array(data["mfccs"])
    y = np.array(data["labels"])
    labels = data["genres"]

    return X, y, labels

def plot_history(history):

    fig, axs = plt.subplots(2, figsize=(10, 8))


    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")


    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="validation loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Evaluation")

    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalize the confusion matrix
    cm = cm * 100  # Convert to percentage
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Load data
    X, y, labels = load_data(dataset)


    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    model = keras.Sequential([
        # Input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),

        # 2nd dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),



        # Output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)


    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100, callbacks=[early_stopping, reduce_lr])


    plot_history(history)


    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, labels)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
