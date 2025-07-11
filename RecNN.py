import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_PATH = "DataMFCC.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfccs"])
    y = np.array(data["labels"])
    labels = data.get("genres", [str(i) for i in range(10)])  # Assuming genres or default genre labels
    return X, y, labels

def plot_history(history):
    fig, axs = plt.subplots(2)
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

def prepare_datasets(test_size, validation_size):
    X, y, labels = load_data(DATA_PATH)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + validation_size)))
    return X_train, X_validation, X_test, y_train, y_validation, y_test, labels

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format=".0f")
    plt.title('Confusion Matrix')
    plt.show()

def build_model(input_shape):


    # build network topology
    model = keras.Sequential()

    # LSTM layers with L2 regularization
    model.add(keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(64, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))

    # Dense layers with L2 regularization
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test, labels = prepare_datasets(0.15, 0.15)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100, callbacks=[early_stopping, reduce_lr])
    plot_history(history)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, labels)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
