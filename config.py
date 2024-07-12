# Konfiguračný súbor config.py:

# Súbor s datasetom
DATASET_PATH = 'dataset.csv'
COLORS = ['red', 'green', 'blue', 'purple']


# Počet bodov v jednej farbe
N = 10000

# Rozdelenie na trénovaciu a testovaciu množinu
TEST_SIZE = 0.1

# Tvorba modelu
##   Input   ->    Dense        ->   Dropout   ->      Dense      ->    Dropout   ->     Dense_3     ->  Output
##    20     ->  DENSE_Hidden_1 ->  DROPOUT_1  -> DENSE_Hidden_2  ->   DROPOUT_2  ->  DENSE_Hidden_3 ->     4

## Vrstvy
DENSE_Hidden_1 = 32
DENSE_Hidden_2 = 32
DENSE_Hidden_3 = 16
## Dropouty
DROPOUT_1 = 0.25
DROPOUT_2 = 0.1
## Learning rate
MODEL_LEARNING_RATE = 0.01

# Trenovanie modelu
## Epochy
EPOCHS = 50
## Batch size
BATCH_SIZE = 32
