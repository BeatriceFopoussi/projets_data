import pandas as pd
import plotly.express as px


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


def variable_to_drop(
    df: pd.DataFrame, 
    verbose: int = 0
) -> pd.DataFrame:
    """
    Supprime les colonnes contenant des valeurs manquantes,
    unimodales,
    inutiles
    et les lignes à valeurs manquantes et inutiles.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        verbose (int, optional): Niveau de détails des informations affichées. Par défaut 0.
    
    Returns:
        pd.DataFrame: Les lignes et colonnes restantes.
    """
    # Filtrer les colonnes avec une seule valeur unique
    columns_with_single_value = df.nunique()[df.nunique() == 1].index.tolist()
    df = df.drop(columns=columns_with_single_value)
    
    # Exclusion des valeurs spécifiques
    values_to_exclude = ['Multifamily MR (5-9)', 'Multifamily LR (1-4)', 'Multifamily HR (10+)']
    df = df[~df['BuildingType'].isin(values_to_exclude)]
    df = df.drop(df[df['DefaultData'] == True].index)
    df = df.drop(df[df['ComplianceStatus'] == "Non-Compliant"].index)
    
    # Suppression des colonnes avec des valeurs manquantes
    df = df.dropna(axis=1, how='all')
    df.dropna(subset=['LargestPropertyUseType', 'LargestPropertyUseTypeGFA'], inplace=True)

    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna(subset=['ENERGYSTARScore', 'SiteEnergyUse(kBtu)'])
    
    # Suppression des colonnes inutiles
    columns_to_drop = [
        "PropertyName",
        'OSEBuildingID',
        'TaxParcelIdentificationNumber',
        'Address',
        'ZipCode',
        'CouncilDistrictCode',
        'NaturalGas(kBtu)',
        'Electricity(kWh)',
        'SiteEUI(kBtu/sf)',
        'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)',
        'SourceEUIWN(kBtu/sf)',
        'SiteEnergyUse(kBtu)',
        'GHGEmissionsIntensity',
        'YearsENERGYSTARCertified',
        
    ]
    df = df.drop(columns=columns_to_drop)
    
    # Affichage des informations si verbose > 0
    if verbose > 0:
        num_cols_before = df.shape[1]
        num_rows_before = df.shape[0]
        print("Nombre de colonnes avant la suppression :", num_cols_before)
        
        df_after = df.dropna(axis=1, how='all')
        df_after = df_after.drop(columns=columns_to_drop)
        
        num_cols_after = df_after.shape[1]
        num_rows_after = df_after.shape[0]
        
        print("Nombre de colonnes après la suppression :", num_cols_after)
        print("Nombre de lignes avant la suppression :", num_rows_before)
        print("Nombre de lignes après la suppression :", num_rows_after)
        print("Proportion de lignes restantes :", num_rows_after / num_rows_before)
        print("Proportion de colonnes restantes :", num_cols_after / num_cols_before)
    
    # Retourne le DataFrame après suppression
    return df

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, max_error
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import matplotlib.pyplot as plt

def random_forest_regression(
    X_train_final, X_test_final, y_train, y_test):
    """
    Effectue la régression avec un modèle de forêt aléatoire.

    Args:
        X_train_final  Les caractéristiques (features) de l'ensemble d'entraînement.
        X_test_final Les caractéristiques (features) de l'ensemble de test.
        y_train : Les valeurs cibles de l'ensemble d'entraînement.
        y_test : Les valeurs cibles de l'ensemble de test.

    Returns:
        tuple: Un tuple contenant:
            - RandomForestRegressor: Le modèle de forêt aléatoire entraîné.
            - array-like: Les prédictions sur l'ensemble de test.
            - array-like: Les prédictions de validation croisée.
            - float: Erreur quadratique moyenne (RMSE).
            - float: Erreur absolue moyenne (MAE).
            - float: Coefficient de détermination (R²).
            - float: Erreur absolue médiane.
            - float: Erreur de prédiction maximale (Max Error).
            - float: Écart type des scores de validation croisée.
            - float: MAE moyenne lors de la validation croisée.
            - float: R² moyen lors de la validation croisée.
    """
        
    # Créer et entraîner le modèle de Random Forest
    forest_regressor = RandomForestRegressor(random_state=42)
    forest_regressor.fit(X_train_final, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = forest_regressor.predict(X_test_final)

    # Calculer les métriques de performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    median_ae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    print("Métriques pour le Random Forest:")
    print("R²:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Median Absolute Error:", median_ae)
    print("Max Error:", max_err)

    # Validation croisée avec R²
    cv_scores = cross_val_score(forest_regressor, X_train_final, y_train, cv=5, scoring='r2')
    print("R² moyen lors de la validation croisée:", cv_scores.mean())

    # Calculer l'écart type après la validation croisée
    std_cv_scores = np.std(cv_scores)
    print("Écart type des scores de validation croisée:", std_cv_scores)

    # Calculer la MAE de la cross-validation
    mae_cv = -cross_val_score(forest_regressor, X_train_final, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    print("MAE moyenne lors de la validation croisée:", mae_cv)

    # Calculer les prédictions avec la validation croisée
    y_cv_pred = cross_val_predict(forest_regressor, X_test_final, y_test, cv=5)

    # Scatter plot pour le Random Forest
    plt.scatter(y_test, y_pred, label='Random Forest', color='blue')
    plt.scatter(y_test, y_cv_pred, label='CV Predictions', color='orange', alpha=0.5) # #     Prédictions de validation croisée
    plt.plot(y_test, y_test, color='red', linestyle='--', label='prédiction')
    plt.xlabel('valeurs')
    plt.xscale("log")
    plt.yscale('log')
    plt.ylabel('Predicted')
    plt.title('Random Forest: Predicted vs Actual')
    plt.legend()

    plt.savefig('randomforest.png')
    plt.show()

    return forest_regressor, y_pred, y_cv_pred, rmse, mae, r2, median_ae, max_err, std_cv_scores, mae_cv, cv_scores.mean()


from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def multiple_regression(
    X_train_final, X_test_final, y_train, y_test):
    """
    Effectue la régression linéaire multiple.

    Args:
        X_train : Les caractéristiques (features) de l'ensemble d'entraînement.
        X_test : Les caractéristiques (features) de l'ensemble de test.
        y_train : Les valeurs cibles de l'ensemble d'entraînement.
        y_test : Les valeurs cibles de l'ensemble de test.

    Returns:
        tuple: Un tuple contenant:
            - LinearRegression: Le modèle de régression linéaire multiple entraîné.
            - array-like: Les prédictions sur l'ensemble de test.
            - float: Erreur quadratique moyenne (RMSE).
            - float: Erreur absolue moyenne (MAE).
            - float: Coefficient de détermination (R²).
    """
        
    # Créer et entraîner le modèle de régression linéaire multiple
    regression_model = LinearRegression()
    regression_model.fit(X_train_final, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = regression_model.predict(X_test_final)

    # Calculer les métriques de performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Visualisation
    plt.scatter(y_test, y_pred, label='Linear Regression', color='yellow')
    plt.plot(y_test, y_test, color='red', linestyle='--', label='prédiction')
    plt.xlabel('valeurs')
    plt.ylabel('Predicted')
    plt.title('Multiple Linear Regression: Predicted ')
    plt.legend()
    plt.savefig('regressionlinéaire.png')

    plt.show()

    return rmse, mae, r2

from sklearn.svm import SVR

def svm_regression(
    X_train_final, X_test_final, y_train, y_test):
    """
    Effectue la régression avec un modèle SVM (Support Vector Machine).

    Args:
        X_train_final : Les caractéristiques (features) de l'ensemble d'entraînement.
        X_test_final : Les caractéristiques (features) de l'ensemble de test.
        y_train : Les valeurs cibles de l'ensemble d'entraînement.
        y_test : Les valeurs cibles de l'ensemble de test.

    Returns:
        tuple: Un tuple contenant:
            - SVR: Le modèle SVM entraîné.
            - array-like: Les prédictions sur l'ensemble de test.
            - array-like: Les prédictions de validation croisée.
            - float: Erreur quadratique moyenne (RMSE).
            - float: Erreur absolue moyenne (MAE).
            - float: Coefficient de détermination (R²).
            - float: Erreur absolue médiane.
            - float: Erreur de prédiction maximale (Max Error).
            - float: Écart type des scores de validation croisée.
            - float: MAE moyenne lors de la validation croisée.
            - float: R² moyen lors de la validation croisée.
    """
    
    # Créer et entraîner le modèle SVM
    svm_regressor = SVR()
    svm_regressor.fit(X_train_final, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = svm_regressor.predict(X_test_final)

    # Calculer les métriques de performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    median_ae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    print("Métriques pour le SVM:")
    #print("R²:", r2)
    #print("MAE:", mae)
    #print("RMSE:", rmse)
    #print("Median Absolute Error:", median_ae)
    #print("Max Error:", max_err)

    # Validation croisée avec R²
    cv_scores = cross_val_score(svm_regressor, X_train_final, y_train, cv=5, scoring='r2')
    print("R² moyen lors de la validation croisée:", cv_scores.mean())

    # Calculer l'écart type après la validation croisée
    std_cv_scores = np.std(cv_scores)
    #print("Écart type des scores de validation croisée:", std_cv_scores)

    # Calculer la MAE de la cross-validation
    mae_cv = -cross_val_score(svm_regressor, X_train_final, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    print("MAE moyenne lors de la validation croisée:", mae_cv)

    # Calculer les prédictions avec la validation croisée
    y_cv_pred = cross_val_predict(svm_regressor, X_test_final, y_test, cv=5)

    # Scatter plot pour le SVM
    plt.scatter(y_test, y_pred, label='SVM', color='green')
    plt.scatter(y_test, y_cv_pred, label='CV Predictions', color='orange', alpha=0.5) #       #Prédictions de validation croisée
    plt.plot(y_test, y_test, color='red', linestyle='--', label='prédiction')
    plt.xlabel('valeurs')
    plt.xscale("log")
    plt.yscale('log')
    plt.ylabel('Predicted')
    plt.title('SVM: Predicted ')
    plt.legend()

    plt.savefig('svm_regression.png')
    plt.show()

    return svm_regressor, y_pred, y_cv_pred, rmse, mae, r2, median_ae, max_err, std_cv_scores, mae_cv, cv_scores.mean()

import xgboost as xgb

def xgboost_regression(X_train_final, X_test_final, y_train, y_test):
    """
    Effectue la régression avec un modèle XGBoost.

    Args:
        X_train_final : Les caractéristiques (features) de l'ensemble d'entraînement.
        X_test_final : Les caractéristiques (features) de l'ensemble de test.
        y_train : Les valeurs cibles de l'ensemble d'entraînement.
        y_test : Les valeurs cibles de l'ensemble de test.

    Returns:
        tuple: Un tuple contenant:
            - XGBRegressor: Le modèle XGBoost entraîné.
            - array-like: Les prédictions sur l'ensemble de test.
            - array-like: Les prédictions de validation croisée.
            - float: Erreur quadratique moyenne (RMSE).
            - float: Erreur absolue moyenne (MAE).
            - float: Coefficient de détermination (R²).
            - float: Erreur absolue médiane.
            - float: Erreur de prédiction maximale (Max Error).
            - float: Écart type des scores de validation croisée.
            - float: MAE moyenne lors de la validation croisée.
            - float: R² moyen lors de la validation croisée.
    """
    
    # Créer et entraîner le modèle XGBoost
    xgb_regressor = xgb.XGBRegressor()
    xgb_regressor.fit(X_train_final, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = xgb_regressor.predict(X_test_final)

    # Calculer les métriques de performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    median_ae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    print("Métriques pour XGBoost:")
    print("R²:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Median Absolute Error:", median_ae)
    print("Max Error:", max_err)

    # Validation croisée avec R²
    cv_scores = cross_val_score(xgb_regressor, X_train_final, y_train, cv=5, scoring='r2')
    print("R² moyen lors de la validation croisée:", cv_scores.mean())

    # Calculer l'écart type après la validation croisée
    std_cv_scores = np.std(cv_scores)
    print("Écart type des scores de validation croisée:", std_cv_scores)

    # Calculer la MAE de la cross-validation
    mae_cv = -cross_val_score(xgb_regressor, X_train_final, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    print("MAE moyenne lors de la validation croisée:", mae_cv)

    # Calculer les prédictions avec la validation croisée
    y_cv_pred = cross_val_predict(xgb_regressor, X_test_final, y_test, cv=5)

    # Scatter plot pour XGBoost
    plt.scatter(y_test, y_pred, label='XGBoost', color='green')
    plt.scatter(y_test, y_cv_pred, label='CV Predictions', color='yellow', alpha=0.5) # Prédictions de validation croisée
    plt.plot(y_test, y_test, color='red', linestyle='--', label='prédiction')
    plt.xlabel('Actual')
    plt.xscale("log")
    plt.yscale('log')
    plt.ylabel('Predicted')
    plt.title('XGBoost: Predicted ')
    plt.legend()

    plt.savefig('xgboost_regression.png')
    plt.show()

    return xgb_regressor, y_pred, y_cv_pred, rmse, mae, r2, median_ae, max_err, std_cv_scores, mae_cv, cv_scores.mean()

import matplotlib.pyplot as plt

#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

def neural_network_regression(X_train_final, X_test_final, y_train, y_test):
    """
    Effectue la régression avec un modèle de réseau de neurones.

    Args:
        X_train_final : Les caractéristiques (features) de l'ensemble d'entraînement.
        X_test_final : Les caractéristiques (features) de l'ensemble de test.
        y_train : Les valeurs cibles de l'ensemble d'entraînement.
        y_test : Les valeurs cibles de l'ensemble de test.

    Returns:
        tuple: Un tuple contenant:
            - Sequential: Le modèle de réseau de neurones entraîné.
            - array-like: Les prédictions sur l'ensemble de test.
            - float: Erreur quadratique moyenne (RMSE).
            - float: Erreur absolue moyenne (MAE).
            - float: Coefficient de détermination (R²).
    """
    
    # Créer le modèle de réseau de neurones
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_final.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    # Entraîner le modèle
    model.fit(X_train_final, y_train, epochs=10, batch_size=32, verbose=0)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test_final).flatten()

    # Calculer les métriques de performance
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("Métriques pour le réseau de neurones:")
    print("R²:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # Visualisation
    plt.scatter(y_test, y_pred, label='Predictions', color='blue')
    plt.plot(y_test, y_test, color='red', linestyle='--', label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Neural Network Regression: Predicted vs Actual')
    plt.legend()
    plt.savefig('neural_network_regression.png')
    plt.show()

    return model, y_pred, rmse, mae, r2





