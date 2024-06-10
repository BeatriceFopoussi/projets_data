import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
#import education as e
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#Modification des affichages de colonnes, lignes et largeurs de colonnes pour avoir un maximum d'information
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 1)



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




def afficher_informations():
    serie_code =df1.groupby('SeriesCode')[['CountryCode']].count().reset_index().sort_values(by='CountryCode')
    nb_pays_par_serie_code =df1.groupby('SeriesCode')[['CountryCode']].count()['CountryCode'].sort_values()

    print("Regroupement des pays par code indicateur:")
    for code, nb_pays in zip(serie_code['SeriesCode'], nb_pays_par_serie_code):
        print(f"Code indicateur: {code}, Nombre de pays: {nb_pays}")

    description = df1.groupby('DESCRIPTION')[['CountryCode']].count().reset_index().sort_values(by='CountryCode').tail(10)
    nb_pays_par_description =df1.groupby('DESCRIPTION')[['CountryCode']].count()['CountryCode'].sort_values().tail(10)

    print("\nRegroupement des pays par description:")
    for desc, nb_pays in zip(description['DESCRIPTION'], nb_pays_par_description):
        print(f"Description: {desc}, Nombre de pays: {nb_pays}")

    print("\nRépartition des pays par région:")
    for region, nb_pays in zip(df2.index, df2.values):
        print(f"Région: {region}, Nombre de pays: {nb_pays}")
        


def info_data(
    df: pd.DataFrame, verbose: int = 0) -> tuple:
    """
    Répartition du nombre de données renseignées par années pour tous les
    indicateurs et tous les pays
    Regroupement du nombre de données par décennie, les années 70, 80...
    Affichage de la répartition des données exploitables par décennie
    Taux de données non nulles par année de la décennie 2010
    Taux de données non nulles par année de la décennie 2010
    Nombre de NaN par année de la décennie 2010
    
    Returns:
        tuple: Un tuple contenant le nombre de données par années,
        le pourcentage de données manquantes et pertinentes
    """
    present = df.loc[:, '1970':'2100'].notnull().sum()
    print("Répartition du nombre de données par années:")
    for year, count in present.items():
        print(f"Année: {year}, Nombre de données: {count}")

    decade = df.loc[:, '1970':'2020'].copy().count()
    ans = ['1970s', '1980s', '1990s', '2000s', '2010s']
    for i in range(5):
        j = i * 10
        k = j + 10
        decade[ans[i]] = decade[j:k, ].sum()

    print("\nNombre de données par décennie:")
    for decade_name, count in decade[ans].items():
        print(f"Décennie: {decade_name}, Nombre de données: {count}")

    annees = ['2010', '2011', '2012', '2013', '2015', '2016', '2017']
    nb_nonnul = df.copy()[annees].count()
    nb_tot = df.copy()[annees].shape[0]
    df_2010s = pd.DataFrame({'annee': nb_nonnul.index, 'nb_nonnul': nb_nonnul.values})

    df_2010s['%_nonnul'] = round((df_2010s['nb_nonnul']) * 100 / nb_tot, 2)

    df_2010s['%_nan'] = round(100 - df_2010s['%_nonnul'], 2)
    print("\nTaux de remplissage entre 2010 et 2017:")
    for year, taux_nonnul, taux_nan in zip(df_2010s['annee'], df_2010s['%_nonnul'], df_2010s['%_nan']):
        print(f"Année: {year}, Taux de données non nulles: {taux_nonnul}%, Taux de NaN: {taux_nan}%")
        
        
def merge_dataframes(Total, df3, df5, df2):
    #Total = df3.copy()

    Total = pd.merge(Total, df5[['Series Code', 'Long definition']], left_on='Indicator Code', right_on='Series Code')

    Total = pd.merge(Total, df2[['Country Code', 'Region']], left_on='Country Code', right_on='Country Code')

    Total = pd.merge(Total, df5[['Series Code', 'Topic']], left_on='Indicator Code', right_on='Series Code')

    Total = pd.merge(Total, df2[['Country Code', 'Income Group']], how='inner', left_on='Country Code', right_on='Country Code')

    if 'Income Group_y' in Total.columns:
        Total.drop('Income Group_y', axis=1, inplace=True)

    Total.drop('Series Code_y', axis=1, inplace=True)
    
    Total.rename(columns={'Series Code_x': 'Series Code', 'Income Group_x': 'Income Group'}, inplace=True)
    Total.drop('Long definition', axis=1, inplace=True)
    Total.drop('Series Code', axis=1, inplace=True)
    Total.drop('Topic', axis=1, inplace=True)
   
    drop = Total.loc[:, '1970':'2009']
    Total = Total.drop(columns=drop.columns)

    return Total

def stat_des(df_num, df_eco, df_edu_sec, df_edu_ter, df_pop):
    # Moyenne de tous les indicateurs
    mean_num = df_num['Année Plus Récente'].mean()
    mean_eco = df_eco['Année Plus Récente'].mean()
    mean_edu_sec = df_edu_sec['Année Plus Récente'].mean()
    mean_edu_ter = df_edu_ter['Année Plus Récente'].mean()
    mean_pop = df_pop['Année Plus Récente'].mean()

    # Médiane de tous les indicateurs
    median_num = df_num['Année Plus Récente'].median()
    median_eco = df_eco['Année Plus Récente'].median()
    median_edu_sec = df_edu_sec['Année Plus Récente'].median()
    median_edu_ter = df_edu_ter['Année Plus Récente'].median()
    median_pop = df_pop['Année Plus Récente'].median()

    # Variance de tous les indicateurs
    var_num = df_num['Année Plus Récente'].var(ddof=0)
    var_eco = df_eco['Année Plus Récente'].var(ddof=0)
    var_edu_sec = df_edu_sec['Année Plus Récente'].var(ddof=0)
    var_edu_ter = df_edu_ter['Année Plus Récente'].var(ddof=0)
    var_pop = df_pop['Année Plus Récente'].var(ddof=0)

    # Écart type de tous les indicateurs
    std_num = df_num['Année Plus Récente'].std(ddof=0)
    std_eco = df_eco['Année Plus Récente'].std(ddof=0)
    std_edu_sec = df_edu_sec['Année Plus Récente'].std(ddof=0)
    std_edu_ter = df_edu_ter['Année Plus Récente'].std(ddof=0)
    std_pop = df_pop['Année Plus Récente'].std(ddof=0)

    # Asymétrie non biaisée de tous les indicateurs
    skew_num = df_num['Année Plus Récente'].skew()
    skew_eco = df_eco['Année Plus Récente'].skew()
    skew_edu_sec = df_edu_sec['Année Plus Récente'].skew()
    skew_edu_ter = df_edu_ter['Année Plus Récente'].skew()
    skew_pop = df_pop['Année Plus Récente'].skew()

    # Aplatissement non biaisé de tous les indicateurs
    kurtosis_num = df_num['Année Plus Récente'].kurtosis()
    kurtosis_eco = df_eco['Année Plus Récente'].kurtosis()
    kurtosis_edu_sec = df_edu_sec['Année Plus Récente'].kurtosis()
    kurtosis_edu_ter = df_edu_ter['Année Plus Récente'].kurtosis()
    kurtosis_pop = df_pop['Année Plus Récente'].kurtosis()

    # Définition d'un dataframe contenant les statistiques
    data_stats = [['mean', mean_num, mean_eco, mean_edu_sec, mean_edu_ter, mean_pop],
                  ['median', median_num, median_eco, median_edu_sec, median_edu_ter, median_pop],
                  ['var', var_num, var_eco, var_edu_sec, var_edu_ter, var_pop],
                  ['std', std_num, std_eco, std_edu_sec, std_edu_ter, std_pop],
                  ['skew', skew_num, skew_eco, skew_edu_sec, skew_edu_ter, skew_pop],
                  ['kurtosis', kurtosis_num, kurtosis_eco, kurtosis_edu_sec, kurtosis_edu_ter, kurtosis_pop]]
    df_stat = pd.DataFrame(data_stats, columns=['Desc', 'Stat_num', 'Stat_eco', 'Stat_edu_sec', 'Stat_edu_ter', 'Stat_pop'])

    return df_stat

import plotly.graph_objects as go
import plotly.subplots as sp

def generate_visualization(year):
    fig = sp.make_subplots(rows=5, cols=2, subplot_titles=("Indicateur démographique", "Indicateur démographique",
                                                          "Indicateur économique", "Indicateur économique",
                                                          "Indicateur éducatif lycée", "Indicateur éducatif lycée",
                                                          "Indicateur éducatif ens. supérieur", " ens. supérieur",
                                                          "Indicateur numérique", "Indicateur numérique"))

    # Indicateur démographique : SP.POP.1524.TO.UN
    df_pop = df[df['Indicator Code'] == 'SP.POP.1524.TO.UN']
    fig.add_trace(go.Box(x=df_pop['Indicator Code'], y=df_pop[year], name='Nombres lycéens et étudiants de 15-24 ans'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_pop[year], nbinsx=10, name='Nombres lycéens et étudiants de 15-24 ans'), row=1, col=2)

    # Indicateur économique : NY.GNP.PCAP.PP.CD
    df_eco = df[df['Indicator Code'] == 'NY.GNP.PCAP.PP.CD']
    fig.add_trace(go.Box(x=df_eco['Indicator Code'], y=df_eco[year], name='Revenu par tête, PPA ($)'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df_eco[year], nbinsx=10, name='Revenu par tête, PPA ($)'), row=2, col=2)

    # Indicateur éducatif lycée : SE.SEC.ENRR
    df_edu_sec = df[df['Indicator Code'] == 'SE.SEC.ENRR']
    fig.add_trace(go.Box(x=df_edu_sec['Indicator Code'], y=df_edu_sec[year], name='Taux scolarisé lycée (%)'), row=3, col=1)
    fig.add_trace(go.Histogram(x=df_edu_sec[year], nbinsx=10, name='Taux scolarisé lycée (%)'), row=3, col=2)

    # Indicateur éducatif ens. supérieur : SE.TER.ENRR
    df_edu_ter = df[df['Indicator Code'] == 'SE.TER.ENRR']
    fig.add_trace(go.Box(x=df_edu_ter['Indicator Code'], y=df_edu_ter[year], name='Taux scolarisé ens. supérieur (%)'), row=4, col=1)
    fig.add_trace(go.Histogram(x=df_edu_ter[year], nbinsx=10, name='Taux scolarisé ens. supérieur (%)'), row=4, col=2)

    # Indicateur numérique : IT.NET.USER.P2
    df_num = df[df['Indicator Code'] == 'IT.NET.USER.P2']
    fig.add_trace(go.Box(x=df_num['Indicator Code'], y=df_num[year], name='Taux d\'utilisateur d\'internet (%)'), row=5, col=1)
    fig.add_trace(go.Histogram(x=df_num[year], nbinsx=10, name='Taux d\'utilisateur d\'internet (%)'), row=5, col=2)

    fig.update_layout(height=1000, width=800, title_text="Graphiques statistiques sur les 7 indicateurs pertinents")
    fig.show()
