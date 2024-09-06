# Moteur de Recherche de Symptômes

Cette application est un moteur de recherche de maladies basé sur les symptômes. Elle utilise des techniques de traitement du langage naturel telles que TF-IDF et la similarité cosinus pour prédire les maladies probables en fonction des symptômes sélectionnés par l'utilisateur. 

## Fonctionnalités

- **Sélection des symptômes :** Sélectionnez les symptômes pertinents pour prédire les maladies potentielles.
- **Profil du patient :** Spécifiez le sexe et la catégorie d'âge pour obtenir des prédictions personnalisées.
- **Prédiction des maladies :** L'application affiche les maladies probables avec un pourcentage de similarité.
- **Graphiques interactifs :** Visualisez la répartition des maladies probables avec des graphiques en camembert.
- **Évaluation des performances :** Affichage de mesures d'évaluation telles que la précision, le rappel et le F1-score.

## Technologies Utilisées

- **Python**
- **Streamlit** : pour l'interface utilisateur
- **Pandas** : pour la gestion des données
- **Scikit-learn** : pour la vectorisation TF-IDF et le calcul de similarité cosinus
- **Plotly** : pour les graphiques interactifs
- **Openpyxl** : pour la gestion des fichiers Excel

## Prérequis

Assurez-vous d'avoir les dépendances Python installées. Les bibliothèques nécessaires sont listées dans le fichier `requirements.txt`.

## Installation

1. Clonez ce dépôt sur votre machine locale :

   ```bash
   git clone https://github.com/username/my-streamlit-app.git
