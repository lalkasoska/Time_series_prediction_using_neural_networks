from funcs import *

menu = {}  # Создаём меню для взаимодействия с программой
menu['1'] = "Status"
menu['2'] = "Predict"
menu['3'] = "Train new model"
menu['4'] = "Save"
menu['5'] = "Load"
menu['6'] = "Exit"

# model, x, y, data_mean, data_std, data_path, model_path, weights_path, past_history, future_target, features_considered, feature_to_predict, STEP = defaultLoad()  # Производим загрузку конфигурации по умолчанию
if __name__ == '__main__':
    try:
        while True:
            print("")
            for entry in menu.items():  # Выводим все опции меню
                print(entry[0], "-", entry[1])
            selection = input("Please Select: ")  # Считываем выбор пользователя
            print("")

            if selection == '1':  # Вывод текущего статуса модели
                print("Current status:")
                status(data_path, model_path, weights_path, past_history, future_target)


            elif selection == '2':  # Прогнозирование заданного интервала
                if model is not None:
                    start = -1
                    while start < 0 or start >= len(
                            x):  # Проверяем, что нам хватает предыдущих значений для предсказания заданного интервала
                        start = int(input(
                            "Input starting index of the interval (integer from " + str(past_history) + " to " + str(
                                len(x) + past_history - 1) + ")\n")) - past_history
                        if start < 0 or start >= len(x):
                            print("Not enough previous records to predict")

                    prediction = predict(model, x[start:start + 1])  # Выполняем предсказание
                    multi_step_plot(
                        x[start:start + 1][0] * data_std[features_considered.index(feature_to_predict)] + data_mean[
                            features_considered.index(feature_to_predict)],
                        y[start:start + 1][0] * data_std[features_considered.index(feature_to_predict)] + data_mean[
                            features_considered.index(feature_to_predict)],
                        prediction * data_std[features_considered.index(feature_to_predict)] + data_mean[
                            features_considered.index(feature_to_predict)],
                        features_considered.index(feature_to_predict),
                        STEP)  # Строим график предсказания и реальных значений

                else:
                    print("Train or load a model first")


            elif selection == '3':  # Выполняем обучение новой модели
                # Вводим все необходимые параметры
                data_path = input("Enter the path to the data file (.csv)\n")
                n = int(input("Enter the number of parameters prediction is based on (must be positive integer)\n"))
                features_considered = []
                for i in range(n):
                    features_considered.append(input("Enter the name of a parameter (its column title in data file)\n"))
                feature_to_predict = input(
                    "Enter the name of the parameter to predict (its column title in data file)\n")
                past_history = int(input(
                    "Enter the number of previous values on which prediction is based (must be positive integer)\n"))
                future_target = int(input("Enter the length of the interval to predict (must be positive integer)\n"))
                train_split = int(input(
                    "Enter the number of records for training (must be positive integer). The rest will be used for validation.\n"))
                STEP = int(input(
                    "Enter the training step size (must be positive integer). This means only 'step'th records will be used for training.\n"))
                epochs = int(input("Enter the number of epochs (must be positive integer).\n"))
                steps_per_epoch = int(
                    input("Enter the number of steps per one epoch (must be positive integer)\n"))
                validation_steps = int(
                    input("Enter the number of validation steps (must be positive integer)\n"))
                buffer_size = int(
                    input("Enter the size of the buffer used for shuffling (must be positive integer)\n"))
                batch_size = int(input("Enter the batch_size (must be positive integer)\n"))

                print("Preparing data...")
                # Подготавиливаем данные, создаём модель
                model, train_data, validation_data, x, y, data_mean, data_std = prepare_for_training(past_history,
                                                                                                     future_target,
                                                                                                     data_path,
                                                                                                     features_considered,
                                                                                                     feature_to_predict,
                                                                                                     train_split, STEP,
                                                                                                     buffer_size,
                                                                                                     batch_size)
                print("Training...")
                # Тренируем модель
                train_history = train(model, train_data, validation_data, epochs, steps_per_epoch, validation_steps)
                print("Training complete")
                # Выводим график потерь при обучении
                plot_train_history(train_history, "Training and validation loss history")


            elif selection == '4':  # Сохраняем модель и веса в заданные файлы
                if model is not None:
                    model_path = input("Enter the path to model file (.json extention will be added automatically)\n")
                    weights_path = input("Enter the path to weights file (.h5 extention will be added automatically)\n")
                    save(model_path, weights_path, model)
                    print("Model saved successfully")
                else:
                    print("Train or load a model first")


            elif selection == '5':  # Загружаем модель из файла и данные для прогноза.
                model_path = input("Enter the path to model file (.json)\n")
                weights_path = input("Enter the path to weights file (.h5)\n")
                data_path = input("Enter the path to data file (.csv)\n")
                past_history = int(input(
                    "Enter the number of previous values on which prediction is based (must be positive integer)\n"))
                STEP = int(
                    input(
                        "Enter the training step size (must be positive integer). This means only 'step'th records were used for training.\n"))
                future_target = int(input("Enter the length of the interval to predict (must be positive integer)\n"))
                n = int(input("Enter the number of parameters prediction is based on (must be positive integer)\n"))
                features_considered = []
                for i in range(n):
                    features_considered.append(input("Enter the name of a parameter (its column title in data file)\n"))
                feature_to_predict = input("Enter the name of the parameter to predict\n")

                model = None
                model = load(model_path, weights_path)
                if model is not None:
                    print("Loading...")
                    df = pd.read_csv(data_path)
                    features = df[features_considered]
                    features.index = df['Date Time']
                    dataset = features.values
                    data_mean = dataset.mean(axis=0)
                    data_std = dataset.std(axis=0)
                    dataset = (dataset - data_mean) / data_std
                    x, y = multivariate_data(dataset, dataset[:, features_considered.index(feature_to_predict)], 0,
                                             None, past_history,
                                             future_target, STEP)
                    print("Model loaded successfully")


            elif selection == '6':  # Выход из программы
                break


            else:  # В случае неверного ввода
                print("Unknown Option Selected!")
    except:
        print("Something went wrong!")
        input()
