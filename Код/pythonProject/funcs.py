import os  # Библиотека для работы с операционной системой

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключаем лишние информационные сообщения tensorflow

import tensorflow as tf  # Самая важная библиотека с инструментами для создания нейронных сетей

import matplotlib as mpl  # Библиотека для построения графиков
import matplotlib.pyplot as plt
import numpy as np  # Библиотека для работы с многомерными массивами

import pandas as pd  # Библиотека для работы с таблицами



mpl.rcParams['figure.figsize'] = (8, 6)  # Задаём параметры окна с графиками
mpl.rcParams['axes.grid'] = True
#tf.random.set_seed(13)  # Для обеспечения воспроизводимости результатов используется функция set_seed
# Задаём переменным значения по умолчанию
model = None
model_path = None
data_path = None
weights_path = None
x = None
y = None


# STEP = 6
past_history = None
future_target = None


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      # Функция для создания списка фрагментов(подинтервалов) переданного набора данных с заданными параметрами
                      target_size, step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)  # Фактически мы получили x и y, где x - вектор, y - одиночное значение


def create_time_steps(length):  # Функция для создания отметок оси абсцисс на графике прогноза
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, index_to_plot, STEP):  # Функция для построения графика прогноза
    plt.figure(figsize=(12, 6))  # Задаём параметры окна с графиком
    num_in = create_time_steps(len(history))  # Абсциссы предыдущих моментов времени
    num_out = len(true_future)  # Абсциссы предсказываемых моментов времени
    plt.plot(num_in, np.array(history[:, index_to_plot]),
             label='History')  # Строим график значений по которым генерируется прогноз значения величины
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',  # Строим график реальных значений величины
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                 label='Predicted Future')  # Строим график спрогнозированных значений
    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(history, title):  # Функция для построения графиков потерь при обучении
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def prepare_for_training(past_history, future_target, data_path, features_considered, feature_to_predict,
                         # Функция для подготовки данных к обучению новой модели
                         TRAIN_SPLIT, STEP, BUFFER_SIZE, BATCH_SIZE):
    df = pd.read_csv(data_path)  # Загружаем данные в таблицу
    features = df[features_considered]  # Выбираем нужные нам столбцы
    features.index = df['Date Time']  # Устанавливаем индексацию по времени
    dataset = features.values  # Создаём набор данных с которыми будем работать
    # Проводим стандартизацию данных
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    print(data_mean, data_std)
    dataset = (dataset - data_mean) / data_std
    print(dataset.mean(axis=0))
    # Создаём набор всех возможных последовательных интервалов значений данных всего набора данных с заданными параметрами
    x, y = multivariate_data(dataset, dataset[:, features_considered.index(feature_to_predict)], 0,
                             None, past_history,
                             future_target, STEP)
    # Создаём набор всех возможных последовательных интервалов с начала набора данных до отметки конца тренировочных данных (TRAIN_SPLIT) с заданными параметрами. Эти данные будут использоваться для обучения.
    x_train, y_train = multivariate_data(dataset, dataset[:, features_considered.index(feature_to_predict)], 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP)
    # Создаём набор всех возможных последовательных интервалов c отметки тренировочных данных (TRAIN_SPLIT) и до конца набора данных с заданными параметрами. Эти данные будут использоваться для проверки.
    x_validation, y_validation = multivariate_data(dataset, dataset[:, features_considered.index(feature_to_predict)],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP)

    # Собираем наши наборы интервалов в новые наборы данных, перемешиваем их с помощью shuffle() и объединяем их в пакеты с помощью batch()
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    validation_data = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    validation_data = validation_data.batch(BATCH_SIZE).repeat()

    # Создаём новую модель нейросети с заданными параметрами. В нашем случае она состоит из двух рекуррентных слоёв c долгой краткосрочной памятью (LSTM) и одного выходного слоя.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32,
                                   return_sequences=True,
                                   input_shape=x_train.shape[-2:]))  # Задаём количество нейронов и форму входных данных для первого слоя
    model.add(tf.keras.layers.LSTM(16, activation='relu')) # Устанавливаем количество нейронов и функцию активации второго слоя
    model.add(tf.keras.layers.Dense(
        future_target))  # Устанавливаем количество нейронов выходного слоя равным длине интервала предсказания, т.е. один нейрон отвечает за одну точку

    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')  # Компилируем новую модель
    return model, train_data, validation_data, x, y, data_mean, data_std


def status(data_path, model_path, weights_path, past_history,
           future_target):  # Функция для отображения текущих параметров модели
    print("Data path =", data_path)
    print("Model path =", model_path)
    print("Weights path =", weights_path)
    print("Previos records required =", past_history)
    print("Length of the interval to predict =", future_target)


def predict(model, data):  # Функция для выполнения прогноза
    return model.predict(data)[0]


def save(model_path, weights_path, model):  # Функция для сохранения модели и весов нейронной сети
    model_json = model.to_json()
    with open(model_path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path + ".h5")


def load(model_path, weights_path):  # Функция для загрузки модели и весов нейронной сети из файлов
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model


def train(model, train_data, validation_data, epochs=10, steps_per_epoch=200,
          validation_steps=50):  # Функция для инициализации обучения нейронной сети
    return model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
                     validation_steps=validation_steps)


def defaultLoad():  # Функция для загрузки модели по умолчанию (использовалась при тестировании, чтобы не вводить данные вручную)
    print("Loading default configuration...")
    model = load("model.json", "model.h5")
    df = pd.read_csv("jena_climate_2009_2016.csv")
    features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
    features = df[features_considered]
    features.index = df['Date Time']
    real_dataset = features.values
    data_mean = real_dataset.mean(axis=0)
    data_std = real_dataset.std(axis=0)
    dataset = (real_dataset - data_mean) / data_std
    STEP = 6
    past_history = 720
    future_target = 72
    x, y = multivariate_data(dataset, dataset[:, 1], 0,
                             None, past_history,
                             future_target, STEP)
    print("Loaded successfully")
    return model, x, y, data_mean, data_std, "jena_climate_2009_2016.csv", "model.json", "model.h5", 720, 72, ['p (mbar)', 'T (degC)', 'rho (g/m**3)'],'T (degC)', 6
