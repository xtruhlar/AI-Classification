import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Načítanie dát zo súboru
def load_data(file):
    # Pandas DataFrame
    data_frame = pd.read_csv(file)

    # Mapovanie farieb na čísla
    class_mapping = {'R': 0, 'G': 1, 'B': 2, 'P': 3}
    data_frame['Colour'] = data_frame['Colour'].map(class_mapping)

    # Vstupné a výstupné hodnoty
    X = data_frame[data_frame.columns[:-1]].values  # Vstupné hodnoty
    Y = data_frame[data_frame.columns[-1]].values  # Výstupné hodnoty

    # Rozdelenie dát na trénovacie a testovacie
    X, Y, XT, YT = train_test_split(X, Y, test_size=config.TEST_SIZE, random_state=42)

    return X, Y, XT, YT


# Vytvorenie modelu
def create_model():
    model = tf.keras.models.Sequential([
        # 3 Dense vrstvy a 2 Dropout vrstvy
        tf.keras.layers.Dense(config.DENSE_Hidden_1, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dropout(config.DROPOUT_1),
        tf.keras.layers.Dense(config.DENSE_Hidden_2, activation='relu'),
        tf.keras.layers.Dropout(config.DROPOUT_2),
        tf.keras.layers.Dense(config.DENSE_Hidden_3, activation='relu'),
        # Výstupná vrstva s 4 neurónmi pre každú farbu
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Kompilácia modelu
    model.compile(
        # Adam optimizer s nastaveným learning rate
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.MODEL_LEARNING_RATE),
        # Funkcia straty
        loss='sparse_categorical_crossentropy',
        # Metrika na vyhodnotenie modelu - presnosť
        metrics=['accuracy']
    )

    return model


# Generovanie bodov
def generate_points(n):
    # Počet bodov v jednej farbe
    n = int(n / 4)

    # Inicializácia prázdnych polí pre body a výsledky
    data = np.empty((0, 2))
    results = np.empty(0)

    Colours = ['R', 'G', 'B', 'P']
    np.random.shuffle(Colours)  # Náhodné poradie farieb

    # Pre každú farbu vygeneruj n bodov
    for Colour in Colours:
        if Colour == 'R':
            # 1% pravdepodobnosť, že sa vygenerujú body mimo rozsahu
            if np.random.rand() < 0.99:
                x = np.random.randint(-5000, 100, n)
                y = np.random.randint(-5000, 100, n)
            else:
                x = np.random.randint(-5000, 5000, n)
                y = np.random.randint(-5000, 5000, n)
            red = np.stack((x, y), axis=1)
            data = np.concatenate((data, red))
            results = np.concatenate((results, np.zeros(n)))

        elif Colour == 'G':
            if np.random.rand() < 0.99:
                x = np.random.randint(-100, 5000, n)
                y = np.random.randint(-5000, 100, n)
            else:
                x = np.random.randint(-5000, 5000, n)
                y = np.random.randint(-5000, 5000, n)
            green = np.stack((x, y), axis=1)
            data = np.concatenate((data, green))
            results = np.concatenate((results, np.ones(n)))

        elif Colour == 'B':
            if np.random.rand() < 0.99:
                x = np.random.randint(-5000, 100, n)
                y = np.random.randint(-100, 5000, n)
            else:
                x = np.random.randint(-5000, 5000, n)
                y = np.random.randint(-5000, 5000, n)
            blue = np.stack((x, y), axis=1)
            data = np.concatenate((data, blue))
            results = np.concatenate((results, 2 * np.ones(n)))

        elif Colour == 'P':
            if np.random.rand() < 0.99:
                x = np.random.randint(-100, 5000, n)
                y = np.random.randint(-100, 5000, n)
            else:
                x = np.random.randint(-5000, 5000, n)
                y = np.random.randint(-5000, 5000, n)
            purple = np.stack((x, y), axis=1)
            data = np.concatenate((data, purple))
            results = np.concatenate((results, 3 * np.ones(n)))

    # Vrátenie vygenerovaných bodov a výsledkov
    return data, results


def main():
    # Načítanie dát
    X, Y, XT, YT = load_data(config.DATASET_PATH)

    # Vytvorenie modelu
    model = create_model()

    # Trenovanie modelu
    model.fit(X, XT, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, validation_data=(Y, YT))

    # Generovanie bodov
    X_New, Colours = generate_points(config.N * 4)

    # Vyhodnotenie modelu
    model.evaluate(X_New, Colours)

    # Predikcia farieb
    predictions = model.predict(X_New)
    # Získanie indexu najvyššej hodnoty
    predictions_l = np.argmax(predictions, axis=1)

    # Vykreslenie trénovacích a testovacích dát
    plt.scatter(X[:, 0], X[:, 1], c=[config.COLORS[i] for i in XT], label='Training data')
    plt.title('Training data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.xlim(-5050, 5050)
    plt.ylim(-5050, 5050)

    plt.scatter(X_New[:, 0], X_New[:, 1], c=[config.COLORS[i] for i in predictions_l], marker='o', label='Test data')
    plt.title('Generated data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    print("\tDavid Truhlar - 120897 - Artificial Intelligence - Assignment 3")
    print("\t\t Neural Network for classification of points in 2D space")

    choice = input("Do you want to configure the model? If no model uses default configuration (y/n): ")
    if choice == 'n':
        import config
        print("Using default configuration")
    else:
        import config
        print("Which parameters do you want to change?")
        print("1. Dataset path")
        print("2. Number of points in each colour")
        print("3. Test size")
        print("4. Learning rate")
        print("5. Number of Epochs and Batch size")

        choice = input("Enter the number of parameter you want to change: ")

        if choice == '1':
            config.DATASET_PATH = input("Enter the path to the dataset: ")
        elif choice == '2':
            config.N = int(input("Enter the number of points in each colour: "))
        elif choice == '3':
            config.TEST_SIZE = float(input("Enter the test size (0.0 - 1.0): "))
        elif choice == '4':
            config.MODEL_LEARNING_RATE = float(input("Enter the learning rate (0.0 - 1.0): "))
        elif choice == '5':
            config.EPOCHS = int(input("Enter the number of epochs: "))
            config.BATCH_SIZE = int(input("Enter the batch size: "))
    main()
