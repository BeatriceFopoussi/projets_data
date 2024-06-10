import pandas as pd
import numpy as np
#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import datetime
from datetime import datetime
import time
#from sklearn.neighbors import KDTree
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

def missing_values(
    df: pd.DataFrame, 
    verbose: int = 0
) -> tuple:
    """
    Calcule la somme des valeurs manquantes pour chaque colonne d'un DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        verbose (int, optional): Niveau de détails des informations affichées. Par défaut 0.
    
    Returns:
        tuple: Un tuple contenant le nombre de valeurs manquantes pour chaque colonne et le nombre de lignes dupliquées.
    """
    total = df.isnull().sum()   
    duplicated = df.duplicated(keep=False).sum()
    size = df.shape
    
    print("Le total des valeurs manquantes est", total.sum()) 
    print("Nombre de lignes dupliquées :", duplicated)
    print("The size of the database is:", size)
    
    return total, duplicated,size

def nan_percentages(
    df: pd.DataFrame, 
    verbose:
    int = 0
) -> tuple:
    """
    Calcule le pourcentage de valeurs NaN dans chaque colonne
    Définit les tranches pour regrouper les pourcentages de valeurs manquantes
    Utilise pd.cut pour attribuer à chaque pourcentage son intervalle correspondant
    Compte le nombre de colonnes dans chaque intervalle
    Crée le graphique avec Plotly

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        verbose (int, optional): Niveau de détails des informations affichées. Par défaut 0.
    
    Returns:
        tuple: Un tuple contenant le nombre de valeurs manquantes pour chaque colonne et le nombre de lignes dupliquées.
    """

    nan_percentages = (df.isna().sum() / len(df)) * 100
    bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    percentage_intervals = pd.cut(nan_percentages, bins=bins, labels=labels)
    counts = percentage_intervals.value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Intervalles de pourcentage de valeurs NaN', 'y': 'Nombre de colonnes'})
    fig.update_layout(title="Répartition des pourcentages de valeurs NaN par colonne", xaxis_tickangle=-45)
    fig.show()

    return nan_percentages, df.duplicated().sum()


def drop_missing(df, seuil):
    seuil_manquant = len(df) * 0.4
    df = df.dropna(axis=1, thresh=seuil_manquant)
    
def analyse_outliers(df, multiplicateur=1.5):
    def detecter_outliers_iqr(serie, multiplicateur=1.5):
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        seuil_bas = q1 - multiplicateur * iqr
        seuil_haut = q3 + multiplicateur * iqr
        outliers = (serie < seuil_bas) | (serie > seuil_haut)
        return outliers, seuil_bas, seuil_haut
    
    colonnes = df.columns
    outliers_df = pd.DataFrame()
    outliers_min = {}
    
    for col in colonnes:
        if np.issubdtype(df[col].dtype, np.number):
            outliers, seuil_bas, seuil_haut = detecter_outliers_iqr(df[col])
            outliers_df[col] = outliers
            outliers_min[col] = df[col][outliers].min()
        
    outliers_count = outliers_df.sum()
    nb_lignes = df.shape[0]
    outliers_percentage = {col: 100 * count / nb_lignes for col, count in outliers_count.items()}
    data = []
    
    for col in colonnes:
        if np.issubdtype(df[col].dtype, np.number):
            count = outliers_count[col]
            percentage = outliers_percentage[col]
            min_outlier = outliers_min[col]
            data.append([col, count, percentage, min_outlier])
        
    df_results = pd.DataFrame(data, columns=['Colonne', 'Nombre d\'outliers', 'Pourcentage', 'Minimum'])
    # Fonction pour formater les nombres en chaîne de caractères avec ou sans chiffres après la virgule
   
   # Définir la colonne "colonne" comme index
    df_results.set_index('Colonne', inplace=True)

    # Enlever le nom de l'index
    df_results.index.name = None
    
    return df_results  
    
    
    
def filters(df):
    rows, columns = df.shape
    print("Nombre de lignes: ", rows)
    print("Nombre de colonnes: ", columns)

    filled_rows = df.dropna(how='all')  
    most_filled_row_index = filled_rows.notna().sum(axis=1).idxmax()  
    most_filled_row = filled_rows.loc[most_filled_row_index]  

    nan_columns = most_filled_row[most_filled_row.isna()].index
    other_rows = filled_rows.drop(most_filled_row_index)

    final_rows = [most_filled_row]

    for col in nan_columns:
        non_null_row = other_rows[other_rows[col].notna()].head(1)
        if not non_null_row.empty:
            final_rows.append(non_null_row.iloc[0])
            other_rows = other_rows.drop(non_null_row.index)

    final_df = pd.DataFrame(final_rows).reset_index(drop=True)

    if len(final_df) < 3:
        remaining_rows_needed = 3 - len(final_df)
        
        # Supprimer les lignes déjà présentes dans final_df en utilisant l'index d'origine
        original_indices = [row.name for row in final_rows]
        df_remaining = df.drop(original_indices, errors='ignore')
        
        sorted_remaining_rows = df_remaining.notna().sum(axis=1).sort_values(ascending=False)
        top_remaining_rows = sorted_remaining_rows.head(remaining_rows_needed).index
        additional_rows = df_remaining.loc[top_remaining_rows]
        final_df = pd.concat([final_df, additional_rows]).reset_index(drop=True)

    return final_df
    
    

def filters2(df, criteres, niveau):
    cluster_mapping = {
        "Apport énergétique": {"Faible": 1, "Moyenne": 0, "Forte": 2},
        "Nutriscore": {"Faible": 1, "Moyenne": 0, "Forte": 2},
        "Taux de gras": {"Faible": 1, "Moyenne": 0, "Forte": 2},
        "Taux de glucide": {"Faible": 0, "Moyenne": 1, "Forte": 2}
    }

    def afficher_correspondance(cluster_mapping, cluster, criteres):
        for crit, niveaux in cluster_mapping.items():
            if crit != criteres:
                for niv, clust in niveaux.items():
                    if clust == cluster:
                        print(f"{crit} : {niv}")

    if criteres in cluster_mapping and niveau in cluster_mapping[criteres]:
        cluster = cluster_mapping[criteres][niveau]
        afficher_correspondance(cluster_mapping, cluster, criteres)
        df_filtré = df[df['cluster'] == cluster]
        return df_filtré
    else:
        print("Critère ou niveau invalide")
        return None

def produits_similaires(df, code):
    # Trouver la ligne correspondante au code donné
    produit = df[df['code'] == code]
    
    # Vérifier si le code existe dans le dataframe
    if len(produit) == 0:
        print("Le code produit n'a pas été trouvé.")
        return
    
    # Récupérer la valeur du cluster pour ce produit
    cluster = produit.iloc[0]['cluster']
    
    # Récupérer la valeur du 'nutrition_grade_fr' pour ce produit
    nutrition_grade = produit.iloc[0]['nutrition_grade_fr']

    # Créer un dictionnaire avec les caractéristiques et leurs valeurs de cluster correspondantes
    caracteristiques = {
        "Apport énergétique": {1: "Faible", 0: "Moyenne", 2: "Forte"},
        "Nutriscore": {1: "Faible", 0: "Moyenne", 2: "Forte"},
        "Taux de gras": {1: "Faible", 0: "Moyenne", 2: "Forte"},
        "Taux de glucide": {0: "Faible", 1: "Moyenne", 2: "Forte"}
    }
  
    # Afficher les informations en fonction du cluster
    for caract, valeurs in caracteristiques.items():
        print(f"{caract} : {valeurs[cluster]}")
    
    # Sélectionner les lignes ayant le même cluster et le même 'nutrition_grade_fr'
    produits_similaires = df[(df['cluster'] == cluster) & (df['nutrition_grade_fr'] == nutrition_grade)]
    
    # Extraire les termes de categories_fr du produit recherché
    if isinstance(produit.iloc[0]['categories_fr'], str):
        categories_fr = set(produit.iloc[0]['categories_fr'].split(','))

    # Filtrer les produits similaires qui contiennent au moins un des termes de packaging
    produits_similaires = produits_similaires[produits_similaires['categories_fr'].apply(lambda x: any(term in str(x).split(',') for term in categories_fr))]
    
    # Ajouter une nouvelle colonne qui compte le nombre de catégories communes
    produits_similaires['categories_common_count'] = produits_similaires['categories_fr'].apply(lambda x: len(categories_fr.intersection(set(str(x).split(',')))))
    
    # Trier le dataframe selon le nombre de catégories communes
    produits_similaires = produits_similaires.sort_values('categories_common_count', ascending=False)
    
    produit_code = df[df['code'] == code]
    display(produit_code)
    
    return produits_similaires 
    
    
ef anova_test(df, numeric_column, categorical_column):
    # Sélection des colonnes pertinentes
    selected_columns = [numeric_column, categorical_column]

    # Création d'un sous-dataframe avec les colonnes sélectionnées
    sub_df = df[selected_columns].dropna()

    # Effectuer le test ANOVA
    groups = [sub_df[sub_df[categorical_column] == category][numeric_column] 
              for category in sub_df[categorical_column].unique()]
    anova_result = f_oneway(*groups)

    # Affichage du résultat du test
    print("Résultat du test ANOVA :")
    print(anova_result)

    # Interprétation du résultat
    alpha = 0.05
    p_value = anova_result.pvalue
    if p_value < alpha:
        print(f"La p-valeur ({p_value:.4f}) est inférieure à alpha ({alpha}): Rejet de l'hypothèse nulle.")
        print(" les moyennes sont différentes.")
    else:
        print(f"La p-valeur ({p_value:.4f}) est supérieure à alpha ({alpha}): Acceptation de l'hypothèse nulle.")
        print(" moyennes ne sont différentes.")

    

from scipy.stats import f_oneway

def anova_test(df, numeric_column, categorical_column):
    # Sélection des colonnes pertinentes
    selected_columns = [numeric_column, categorical_column]

    # Création d'un sous-dataframe avec les colonnes sélectionnées
    sub_df = df[selected_columns].dropna()

    # Effectuer le test ANOVA
    groups = [sub_df[sub_df[categorical_column] == category][numeric_column] 
              for category in sub_df[categorical_column].unique()]
    anova_result = f_oneway(*groups)

    # Affichage du résultat du test
    print("Résultat du test ANOVA :")
    print(anova_result)

    # Interprétation du résultat
    alpha = 0.05
    p_value = anova_result.pvalue
    if p_value < alpha:
        print(f"La p-valeur ({p_value:.4f}) est inférieure à alpha ({alpha}): Rejet de l'hypothèse nulle.")
        print(" les moyennes sont différentes.")
    else:
        print(f"La p-valeur ({p_value:.4f}) est supérieure à alpha ({alpha}): Acceptation de l'hypothèse nulle.")
        print(" moyennes ne sont différentes.")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    