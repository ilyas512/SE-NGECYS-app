import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px

# Afficher le logo en haut de l'application
st.image('INGECYS.png', width=100)  # Ajustez la largeur selon vos besoins

# Charger les données pour l'affichage
file_path_display = 'symtoms_separated.xlsx'  # Fichier pour afficher et illustrer les données
file_path_tfidf = 'spaces_removed1.xlsx'  # Fichier pour le modèle TF-IDF avec underscores

try:
    # Charger les données pour l'affichage
    data_display = pd.read_excel(file_path_display)
    # Charger les données pour le modèle TF-IDF
    data_tfidf = pd.read_excel(file_path_tfidf)
except Exception as e:
    st.error(f"Erreur lors de la lecture des fichiers Excel : {e}")
    st.stop()

# Préparer les données pour l'affichage
symptome_columns = [f'symptome{i}' for i in range(1, 19)]  # Colonnes de symptome1 à symptome18
data_display['all_symptoms'] = data_display[symptome_columns].fillna('').agg(' '.join, axis=1)

# Combiner les colonnes de symptômes en une seule chaîne par maladie pour le modèle TF-IDF
data_tfidf['all_symptoms'] = data_tfidf[symptome_columns].fillna('').agg(' '.join, axis=1)

# Appliquer TF-IDF sur les données préparées pour le modèle sans remplacer les underscores
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=1)
tfidf_matrix = tfidf.fit_transform(data_tfidf['all_symptoms'])  # Pas de remplacement d'underscores

# Extraire les symptômes uniques des colonnes pour l'affichage
symptomes = set()  # Utiliser un ensemble pour éviter les doublons
for col in symptome_columns:
    if col in data_display.columns:
        symptomes.update(data_display[col].dropna().str.strip().str.lower().unique())  # Ajouter les symptômes uniques après suppression des NaN

# Convertir l'ensemble en liste et trier pour l'affichage
symptoms = sorted(list(symptomes))

# Mise en page de l'application Streamlit
st.title("Moteur de Recherche de Symptômes")

# Ajouter des champs pour le profil du patient
st.subheader("Profil du Patient")

# Utiliser des boutons radio stylisés comme des boutons
sex = st.radio("Sexe :", options=["Homme", "Femme"], horizontal=True)

# Choisir la catégorie d'âge
age_category = st.selectbox("Catégorie d'âge :", options=["Nourrisson", "Enfant", "Adulte", "Personne âgée"])

# Afficher les symptômes correspondants sous forme de suggestions dans un menu déroulant à sélection multiple
selected_symptoms = st.multiselect(
    "Sélectionnez les symptômes correspondants :",
    options=symptoms,
    default=None
)

# Listes des maladies spécifiques aux femmes et aux hommes
female_specific_diseases = [
    "Endométriose", "Cancer du sein", "Cancer de l’ovaire", "Cancer du col de l’utérus",
    "Diabète gestationnel", "Incontinence urinaire", "Ménopause", "Troubles menstruels"
]

male_specific_diseases = [
    "Hypertrophie bénigne de la prostate", "Cancer de la prostate", "Infarctus du myocarde",
    "Troubles de l’érection", "Goutte", "Hémochromatose"
]

# Ajouter des listes de maladies spécifiques par catégorie d'âge
maladies_nourrisson = [
    "Rougeole", 
    "Coqueluche"
]

maladies_enfant = [
    "Hyperactivité et trouble de l’attention chez l’enfant",
    "Rhinopharyngite de l'enfant", 
    "Enurésie (pipi au lit)",
    "Troubles du sommeil chez l’enfant",
    "Diarrhée et gastro-entérite chez l'enfant", 
    "Constipation de bébé et de l'enfant", 
    "Angine et mal de gorge de l’enfant", 
    "Autisme et TED", 
    "Dépression chez l’enfant et l’adolescent", 
    "Douleur chez l’enfant", 
    "Fièvre de l'enfant", 
    "Mal de ventre chez l’enfant", 
    "Nausées et vomissement de l'enfant", 
    "Pneumonie", 
    "Problèmes de peau chez les enfants"
]

maladies_adulte = [
    "Angine et mal de gorge de l’adulte", 
    "Constipation de l'adulte", 
    "Dépression de l'adulte", 
    "Diarrhée et gastro-entérite de l'adulte", 
    "Douleur chez l'adulte", 
    "Enrouement de l'adulte", 
    "Fièvre de l’adulte", 
    "Mal de ventre chez l'adulte", 
    "Nausées et vomissement de l'adulte", 
    "Otite et douleur d’oreille de l'adulte", 
    "Toux chez l'adulte"
]

maladies_personne_agee = [
    "Maladie d'Alzheimer", 
    "Troubles du rythme cardiaque",
    "Dégénérescence maculaire (DMLA)", 
    "Glaucome", 
    "Accident vasculaire cérébral (AVC)"
]

# Fonction pour prédire la maladie en fonction des symptômes sélectionnés et du profil du patient
def predire_maladie(symptomes_selectionnes, sexe, categorie_age):
    if not symptomes_selectionnes:
        return "Aucun symptôme sélectionné. Impossible de prédire."
    
    # Transformer les symptômes sélectionnés pour la prédiction
    nouvelle_vecteur_symptome = tfidf.transform([' '.join(symptomes_selectionnes)])
    
    # Calculer la similarité cosinus
    similarites = cosine_similarity(nouvelle_vecteur_symptome, tfidf_matrix).flatten()
    
    # Obtenir les indices des 10 maladies les plus similaires
    top_10_indices = similarites.argsort()[-10:][::-1]
    top_10_maladies = data_tfidf['NOM'].iloc[top_10_indices]
    top_10_similarites = similarites[top_10_indices] * 100
    
    # Filtrer les maladies selon le sexe
    if sexe == "Homme":
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in zip(top_10_maladies, top_10_similarites)
                             if maladie not in female_specific_diseases]
    else:
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in zip(top_10_maladies, top_10_similarites)
                             if maladie not in male_specific_diseases]
    
    # Filtrer les maladies selon la catégorie d'âge avec exclusions des autres catégories
    if categorie_age == "Nourrisson":
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in maladies_filtrees
                             if maladie not in maladies_adulte and maladie not in maladies_personne_agee
                             and maladie not in maladies_enfant]
    elif categorie_age == "Enfant":
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in maladies_filtrees
                             if maladie not in maladies_nourrisson and maladie not in maladies_adulte 
                             and maladie not in maladies_personne_agee]
    elif categorie_age == "Adulte":
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in maladies_filtrees
                             if maladie not in maladies_nourrisson and maladie not in maladies_enfant 
                             and maladie not in maladies_personne_agee]
    elif categorie_age == "Personne âgée":
        maladies_filtrees = [(maladie, similarite) for maladie, similarite in maladies_filtrees
                             if maladie not in maladies_nourrisson and maladie not in maladies_enfant 
                             and maladie not in maladies_adulte]
    
    # Sélectionner les 5 meilleures maladies après les filtres
    maladies_filtrees = maladies_filtrees[:5]
    
    # Préparer les résultats
    resultats = [f"Profil du patient : {sexe}, {categorie_age}"]
    for i, (maladie, similarite) in enumerate(maladies_filtrees):
        resultats.append(f"Maladie probable {i+1} : {maladie} avec {similarite:.2f}% de similarité \n")
    
    # Extraire les noms des maladies et leurs similarités pour le graphique
    noms_maladies_filtrees = [maladie for maladie, _ in maladies_filtrees]
    similarites_filtrees = [similarite for _, similarite in maladies_filtrees]
    
    # Retourner les résultats et les données pour le graphique
    return '\n'.join(resultats), noms_maladies_filtrees, similarites_filtrees

# Fonction pour afficher les métriques d'évaluation
def afficher_mesures_evaluation(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    st.subheader("Mesures d'Évaluation")
    st.write(f"Précision : {precision:.2f}")
    st.write(f"Rappel : {recall:.2f}")
    st.write(f"F1-Score : {f1:.2f}")

# Fonction pour simuler les résultats d'évaluation (pour illustration)
def evaluer_simulation():
    # Simuler des valeurs de vérité terrain et des prédictions pour l'exemple
    y_true = [1, 0, 1, 0, 1]  # Étiquettes simulées (à remplacer par vos données réelles)
    y_pred = [1, 0, 0, 0, 1]  # Prédictions simulées
    afficher_mesures_evaluation(y_true, y_pred)

# Bouton pour déclencher la prédiction de la maladie
if st.button("Prédire la Maladie"):
    if selected_symptoms:
        # Prédire la maladie en fonction des symptômes sélectionnés et du profil du patient
        prediction, diseases, similarities = predire_maladie(selected_symptoms, sex, age_category)
        
        # Tracé du graphique interactif en histogramme coloré avec Plotly
        fig = px.bar(
            x=diseases,
            y=similarities,
            title="Répartition des Maladies Probables",
            labels={'x': 'Maladies', 'y': 'Similarité (%)'},
            color=diseases,  # Ajouter des couleurs différentes pour chaque maladie
            color_discrete_sequence=px.colors.qualitative.Bold  # Choisir une palette de couleurs
        )
        
        # Mettre à jour la mise en page pour ajuster l'apparence du graphique
        fig.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Maladies",
            yaxis_title="Similarité (%)"
        )
        
        # Améliorer le style des barres et du texte dans le graphique
        fig.update_traces(
            texttemplate='%{y:.2f}%',  # Afficher les pourcentages sur les barres
            textposition='outside'  # Positionner le texte à l'extérieur des barres
        )
        
        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Afficher les détails de la prédiction
        st.write(prediction)

        # Simuler et afficher les mesures d'évaluation
        evaluer_simulation()

    else:
        st.error("Aucun symptôme valide sélectionné. Impossible de prédire.")
