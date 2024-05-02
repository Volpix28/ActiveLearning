import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Pfad zum Ordner mit den Bilddaten
data_dir = 'Pfad_zum_Bilddatenordner'

# Liste der Klassen (Unterordner)
classes = ['Hunde', 'Katzen']

# TODO: Implementieren Sie die Funktion zum Laden der Bilddaten

# TODO: Implementieren Sie die Funktion zum Aufteilen der Daten in Trainings- und Testdaten

# TODO: Implementieren Sie die Funktion zum Training des Modells

# TODO: Implementieren Sie die Funktion für Uncertainty Sampling

# TODO: Implementieren Sie die Funktion für Diversity Sampling

# Laden der Bilddaten
X, y = load_data()

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training des Modells auf den vorhandenen Trainingsdaten
model = train_model(X_train, y_train)

# Aktives Lernen
num_iterations = 5
num_samples_per_iteration = 10

for i in range(num_iterations):
    # TODO: Aktives Lernen mit Uncertainty Sampling und Diversity Sampling
    # - Verwenden Sie die Funktionen für Uncertainty Sampling und Diversity Sampling
    # - Aktualisieren Sie die Trainingsdaten
    # - Trainieren Sie das Modell
    # - Berechnen Sie die Genauigkeit auf den Testdaten
    # - Visualisieren Sie die ausgewählten Datenpunkte
    
    # Ausgabe der Genauigkeit auf den Testdaten
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Iteration {i+1}: Genauigkeit = {accuracy:.4f}")