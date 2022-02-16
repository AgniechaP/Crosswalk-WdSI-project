import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas

def load_data(path, filename):
    """
    Ladowanie danych z folderu, w ktorym znajduja się pliki: Train, Test, Train.csv, Test.csv
    @param path: Sciezka do pliku.
    @param filename: Plik rozszerzenia .csv (Train.csv oraz Test.csv).
    @return: Dane.
    """
    entry_list = pandas.read_csv(os.path.join(path, filename))

    data = []
    for idx, entry in entry_list.iterrows():
        class_id = entry['ClassId']
        image_path = entry['Path']

        if class_id != -1:
            image = cv2.imread(os.path.join(path, image_path))
            data.append({'image': image, 'label': class_id})

    return data

def display_dataset_stats(data):
    """
    Wyswietlanie statystyki dotyczacej danych w plikach, format: class_id: liczba samples
    @param data: Lista danych.
    @return: Nic nie zwraca
    """
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    # print('number of samples for each class:')
    print(class_to_num)

def learn_bovw(data):
    """
    Uczenie BoVW slownika i zapis jako plik "voc.npy".
    @param data: Lisa danych.
    @return: Nic nie zwraca.
    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

def extract_features(data):
    """
    Funkcja implementujaca ekstrakcje cech oraz zapis deskryptorow.
    @param data: Lista danych.
    @return: Dane z dodanymi deskryptorami.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        # compute descriptor and add it as "desc" entry in sample
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)  # robienie deskryptora
        sample['desc'] = desc

    return data

def train(data):  # tylko trenujemy model tutaj
    """
    Trenowanie modelu.
    @return: Wytrenowany model.
    """
    descs = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))  # squeeze zmienia macierz na wektor wokol konkretnej osi -> tu wokol osi 0 (a mozemy wokol 0, 1 lub 2)
            labels.append(sample['label'])
    rf = RandomForestClassifier()
    rf.fit(descs, labels)

    return rf  # wyjsciem funkcji jest model

def predict(rf, data):  # przyjmuje rf gdzie mamy zapisany model i dane porzednie
    """
    Predykuje etykiety i zapisuje je do "label_pred" (int) dla kazdej sample.
    @param rf: Wytrenowany model.
    @param data: Lista danych.
    @return: Dane z dodanymi wypredykowanymi etykietami.
    """

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])  # ta linia jest kluczowa dla predykcji, ale my chcemy zewaluowac cala baze danych dlatego robimy inne linijki
            sample['label_pred'] = int(pred)
    # zwraca etykiete do pred i uzupelniamy tabele data etykietą (etykiety byly 1, 2 ,3)
    # ------------------

    return data  # dane z wypredykowanymi etykietami

def evaluate(data):  # porownanie statystyczne, kolumna label_pred - wypredkowane labele, a w kolumnie label - etykiety prawdziwe. Wykorzystujemy jedna z metryk ewaluacji
    """
    Ewaluacja rezultatow klasyfikacji. Dokladnosc - ile obiektow poprawnie zaklasyfikowano.
    @param data: Lista danych.
    @return: Nic nie zwraca.
    """
    n_corr = 0
    n_incorr = 0
    pred_labels = []
    true_labels = []
    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            pred_labels.append(sample['label_pred'])
            true_labels.append(sample['label'])
            if sample['label_pred'] == sample['label']:
                n_corr += 1
            else:
                n_incorr += 1
    n = n_corr / max(n_corr + n_incorr, 1)
    print("Score = " + str(n))

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(conf_matrix)

    return

def main():
    data_train = load_data('./', 'Train.csv')
    print('Zbior danych treningowych')
    display_dataset_stats(data_train)

    data_test = load_data('./', 'Test.csv')
    print('Zbior danych testowych:')
    display_dataset_stats(data_test)

    # you can comment those lines after dictionary is learned and saved to disk.
    #print('learning BoVW')
    #learn_bovw(data_train)

    print('Ekstrakcja trenowanych cech')
    data_train = extract_features(data_train)

    print('Trenowanie')
    rf = train(data_train)

    print('extracting test features')
    data_test = extract_features(data_test)

    print('testing on testing dataset')
    data_test = predict(rf, data_test)

    evaluate(data_test)


if __name__ == '__main__':
    main()