from IPython.display import display, Markdown
import pandas as pd                    ## Este proporciona una estructura similiar a los data.frame
import statsmodels.api as sm           ## Este proporciona funciones para la estimación de muchos modelos estadísticos
import statsmodels.formula.api as smf  ## Permite ajustar modelos estadísticos utilizando fórmulas de estilo R

import warnings
warnings.filterwarnings("ignore")

# loading packages
# basic + dates 
import numpy as np
from numpy import inf
import pandas as pd
import datetime
import holidays
from pandas import datetime

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# linear regression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

# prophet by Facebook
from fbprophet import Prophet


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from pandas_profiling import ProfileReport #Herramienta para Análisis Descriptivo

from google.cloud import storage #Permite acceder por medio de API a Google Cloud Storage
from google.cloud import bigquery #Permite acceder y hacer consultas SQL a DWH de Google Cloud 
from pandas.io import gbq
import pandas_gbq
from google.oauth2 import service_account
# %load_ext google.cloud.bigquery

from sklearn.metrics import mean_squared_error

from IPython.core.display import display, HTML
display(HTML("<style>.container{width:100% !important;}</style>"))

from sklearn.linear_model import LogisticRegression #Importamos el modelo Regresión Logística
from sklearn.neural_network import MLPRegressor #Importamos el modelo de Redes Neuronales Regresivas
# from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier #Importamos el modelo de Random Forest para problemas de Clasificación
from sklearn.metrics import confusion_matrix #Nos permite visualizar una matriz de confusión
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error #Nos permite obtener el MAE de un modelo
from sklearn.metrics import r2_score #Nos permite obtener el r2 de un modelo
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor
import scikitplot as skplt #Nos permite visualizar un poco más fácil la matriz de confusión
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
from sklearn.neural_network import MLPClassifier

from pandas.io import gbq
import pandas_gbq
from google.oauth2 import service_account

from sklearn.preprocessing import LabelEncoder
sns.set_style("darkgrid")
sns.set_context("notebook")


# A estas alturas, no tenemos la tabla de datos creada, así que haré una llamada a un fichero CSV
# Igualmente, dejaré una función definida para la importación de los datos del DWH de GCP
# La vuelta que tengo que darle es cómo hacer para que cualquier ordenador pueda ejecutar el fichero JSON y pueda acceder sin problemas

def connection_Google_Cloud_Bigquery(json_route,QUERY):
    client=bigquery.Client.from_service_account_json(json_route)
    query_job=client.query(QUERY).result().to_dataframe()
    return query_job

def IMPORTACION_DATOS_MANUAL():

    RESULTS_DF=pd.read_csv(r'C:\Users\victo\Documents\TFM\datasets_584887_1375686_results.csv')
    RESULTS_DF=RESULTS_DF.rename(columns={'FTHG':'HOMEGOALS','FTAG':'AWAYGOALS','FTR':'RESULT'})
    RESULTS_DF['GOALSDIFF']=abs(RESULTS_DF['HOMEGOALS']-RESULTS_DF['AWAYGOALS'])
    RESULTS_DF=RESULTS_DF[['Season','Date','HomeTeam','AwayTeam','HOMEGOALS','AWAYGOALS','GOALSDIFF','RESULT']]
    RESULTS_DF=RESULTS_DF[(RESULTS_DF['Date']>'2010-12-31')&(RESULTS_DF['Season']>='2011-12')]  
    RESULTS_DF=RESULTS_DF.sort_values(['Date'])

    return RESULTS_DF

def IMPORTACION_DATOS_GOOGLE_CLOUD():
    Query=('SELECT*FROM `FOOTBALL_PREDICTOR.PL_HISTORIC`')
    route='My Project 59792-523c1ed8c01a.json'
    RESULTS_DF=connection_Google_Cloud_Bigquery(route,Query)
    RESULTS_DF=RESULTS_DF.rename(columns={'FTHG':'HOMEGOALS','FTAG':'AWAYGOALS','FTR':'RESULT'})
    RESULTS_DF['GOALSDIFF']=abs(RESULTS_DF['HOMEGOALS']-RESULTS_DF['AWAYGOALS'])
    RESULTS_DF=RESULTS_DF[['Season','Date','HomeTeam','AwayTeam','HOMEGOALS','AWAYGOALS','GOALSDIFF','RESULT']]
    RESULTS_DF=RESULTS_DF[(RESULTS_DF['Date']>'2010-12-31')&(RESULTS_DF['Season']>='2011-12')]
    RESULTS_DF=RESULTS_DF.sort_values(['Date'])

    return RESULTS_DF

# El siguiente script, es para que se cree un listado de los equipos que tenemos en nuestro dataset
# Esto lo he tenido que hacer para garantizar que se le asigne a cada equipo un número de identificación único
# Ya que la forma de calcular los ranking de ELO dependen netamente de esta manera de identificación numérica

def LISTADO_NUMERICO_EQUIPOS(DATASET):
    EQUIPOS=pd.DataFrame(DATASET.sort_values('HomeTeam',ignore_index=True).HomeTeam.unique(),columns=['TEAMS']).reset_index()
    EQUIPOS=EQUIPOS.rename(columns={'index':'TEAM_CODE'})

    return EQUIPOS

def LIMPIEZA_DATOS_EQUIPOS_NUMERICOS(DATASET,EQUIPOS_DATASET):
    DATASET['TEAM_A']=0
    DATASET['TEAM_B']=0
    for team in EQUIPOS_DATASET.TEAMS.unique():
        DATASET.loc[DATASET['HomeTeam']==team,'TEAM_A']=int(EQUIPOS_DATASET.loc[EQUIPOS_DATASET['TEAMS']==team,'TEAM_CODE'])
        DATASET.loc[DATASET['AwayTeam']==team,'TEAM_B']=int(EQUIPOS_DATASET.loc[EQUIPOS_DATASET['TEAMS']==team,'TEAM_CODE'])
    
    return DATASET

def CALCULO_RANKING_ELO(DATASET,EQUIPOS_DATASET):
    mean_elo=1500
    elo_width=400
    k_factor=64
    elo_per_season = {}
    n_teams = len(EQUIPOS_DATASET.TEAMS.unique())
    current_elos=np.ones(shape=(n_teams))*mean_elo
    df_team_elos = pd.DataFrame(index=DATASET.Date[DATASET['Season']!='2019-20'].unique(), columns=range(n_teams))
    df_team_elos.iloc[0, :] = current_elos
    def update_elo(winner_elo, loser_elo,goals):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        expected_win = expected_result(winner_elo, loser_elo)
        if goals <=1:
            change_in_elo = k_factor * 1 * (1-expected_win)
        elif goals ==2:
            change_in_elo=k_factor*3/2*(1-expected_win)
        elif goals >= 3:
            change_in_elo = k_factor * ((11+goals)/8) * (1-expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        
        return winner_elo, loser_elo

    def expected_result(elo_a, elo_b):
        """
        https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
        """
        expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
        
        return expect_a

    for row in DATASET.itertuples():
        idx=row.Index
        winner=row.RESULT
        if winner == 'A':
            w_id=row.TEAM_B
            l_id=row.TEAM_A
        else:
            w_id=row.TEAM_A
            l_id=row.TEAM_B
        w_elo_before=current_elos[w_id]
        l_elo_before=current_elos[l_id]
        dif_goals=row.GOALSDIFF
        w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before,dif_goals)
        DATASET.at[idx, 'w_elo_before_game'] = w_elo_before
        DATASET.at[idx, 'l_elo_before_game'] = l_elo_before
        DATASET.at[idx, 'w_elo_after_game'] = w_elo_after
        DATASET.at[idx, 'l_elo_after_game'] = l_elo_after
        current_elos[w_id] = w_elo_after
        current_elos[l_id] = l_elo_after
        today = row.Date
        df_team_elos.at[today, w_id] = w_elo_after
        df_team_elos.at[today, l_id] = l_elo_after
        if winner == 'A':
            DATASET.at[idx,'AWAY_TEAM_ELO_BEFORE_GAME']=w_elo_before
            DATASET.at[idx,'AWAY_TEAM_ELO_AFTER_GAME']=w_elo_after
            DATASET.at[idx,'HOME_TEAM_ELO_BEFORE_GAME']=l_elo_before
            DATASET.at[idx,'HOME_TEAM_ELO_AFTER_GAME']=l_elo_after
        else:
            DATASET.at[idx,'HOME_TEAM_ELO_BEFORE_GAME']=w_elo_before
            DATASET.at[idx,'HOME_TEAM_ELO_AFTER_GAME']=w_elo_after
            DATASET.at[idx,'AWAY_TEAM_ELO_BEFORE_GAME']=l_elo_before
            DATASET.at[idx,'AWAY_TEAM_ELO_AFTER_GAME']=l_elo_after
    
    DATASET.drop(columns={'w_elo_before_game','l_elo_before_game','w_elo_after_game','l_elo_after_game'},inplace=True)
    
    return DATASET,current_elos,df_team_elos

def CLASSIFICATION_MODELS(DATAFRAME_ENTRENAR,DATAFRAME_JUEGOS=False):

        MODELS=['LogisticRegression','RandomForestClassifier','LinearDiscriminantAnalysis','MLPClassifier','DecisionTreeClassifier']
        MAE_list=[]
        Accuracy_list=[]
        RMSE_list=[]
        ROC_AUC_list=[]
        team_order=[]
        confusion_df=pd.DataFrame()
        confusion_norm_df=pd.DataFrame()
        matriz_confusion_df=pd.DataFrame()
        models_list=[]

        for MODEL in MODELS:
            print('Ejecutaremos el modelo '+MODEL)
            for team in DATAFRAME_ENTRENAR['HomeTeam'].unique():
                print('Procederemos a hacer el modelo '+MODEL+' para ',team)
                df_concat_local=DATAFRAME_ENTRENAR[(DATAFRAME_ENTRENAR['HomeTeam']==team)^(DATAFRAME_ENTRENAR['AwayTeam']==team)]
                df_concat_local=df_concat_local[df_concat_local['Season']<'2019-20']
                df_concat_local=df_concat_local[['Season','Date','HomeTeam','AwayTeam','RESULT','GOALSDIFF','TEAM_A','TEAM_B','HOME_TEAM_ELO_BEFORE_GAME','AWAY_TEAM_ELO_BEFORE_GAME']]
                df_concat_local['Ranking_ELO_'+team]=0
                df_concat_local.loc[(df_concat_local.HomeTeam==team),'Ranking_ELO_'+team]=df_concat_local.HOME_TEAM_ELO_BEFORE_GAME
                df_concat_local.loc[(df_concat_local.AwayTeam==team),'Ranking_ELO_'+team]=df_concat_local.AWAY_TEAM_ELO_BEFORE_GAME
                df_concat_local['Ranking_ELO_RESTO_EQUIPOS']=0
                df_concat_local.loc[(df_concat_local.HomeTeam!=team),'Ranking_ELO_RESTO_EQUIPOS']=df_concat_local.HOME_TEAM_ELO_BEFORE_GAME
                df_concat_local.loc[(df_concat_local.AwayTeam!=team),'Ranking_ELO_RESTO_EQUIPOS']=df_concat_local.AWAY_TEAM_ELO_BEFORE_GAME
                df_concat_local['Resultado_'+team]=0
                df_concat_local.loc[(df_concat_local.RESULT=='H')&(df_concat_local.HomeTeam==team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=2
                df_concat_local.loc[(df_concat_local.RESULT=='A')&(df_concat_local.AwayTeam==team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=2
                df_concat_local.loc[(df_concat_local.GOALSDIFF==0),'Resultado_'+team]=1
                df_concat_local.loc[(df_concat_local.RESULT=='H')&(df_concat_local.HomeTeam!=team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=0
                df_concat_local.loc[(df_concat_local.RESULT=='A')&(df_concat_local.AwayTeam!=team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=0
                X=df_concat_local[['Ranking_ELO_RESTO_EQUIPOS','Ranking_ELO_'+team]]
                y=df_concat_local['Resultado_'+team]
                if len(df_concat_local.index) == 0:
                    pass
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                if MODEL == 'LogisticRegression':
                    print('A')
                    clf= LogisticRegression()
                elif MODEL == 'RandomForestClassifier':
                    print('B')
                    clf = RandomForestClassifier(max_depth=10, random_state=0)
                elif MODEL == 'LinearDiscriminantAnalysis':
                    print('C')
                    clf = LinearDiscriminantAnalysis()
                elif MODEL == 'MLPClassifier':
                    print('D')
                    clf = MLPClassifier()
                elif MODEL == 'DecisionTreeClassifier':
                    print('E')
                    clf = DecisionTreeClassifier()
                if MODEL == 'MLPClassifier':
                    scaler = StandardScaler()  # doctest: +SKIP
                    scaler.fit(X_train)  # doctest: +SKIP
                    X_train = scaler.transform(X_train)  
                    X_test = scaler.transform(X_test)
                model=clf.fit(X_train, y_train)
                y_predicted = model.predict(X_test)
                y_predicted_team=y_predicted.copy()
                y_test_team=np.array(y_test.copy())
#                 matriz_confusion_df['y_predicted_'+team]=y_predicted_team
#                 matriz_confusion_df['y_test_'+team]=y_test_team
                team_order.append(team)
                models_list.append(MODEL)
                RMSE=mean_squared_error(y_test,y_predicted)
                RMSE_list.append(RMSE)
                MAE=mean_absolute_error(y_test,y_predicted)
                MAE_list.append(MAE)
                Accuracy=accuracy_score(y_test,y_predicted)
                Accuracy_list.append(Accuracy)
#                 try:
#                     ROC_AUC=roc_auc_score(y_test,y_predicted,multi_class=False)
#                 except:
#                     pass
#                 ROC_AUC_list.append(ROC_AUC)
#                 try:
#                     confusion=pd.DataFrame(confusion_matrix(y_test,y_predicted),columns=['0','1'])
#                     confusion['Case']=['0','1']
#                     confusion['Team']=[team,team]
#                 except:
#                     confusion=pd.DataFrame(confusion_matrix(y_test,y_predicted),columns=['1'])
#                     confusion['0']=0
#                     confusion['Case']=['1']
#                     confusion['Team']=[team]
#                 confusion=confusion[['Team','Case','0','1']]
#                 confusion.append([team,'0',0,0])
#                 confusion=confusion[['Team','Case','0','1']]
#                 confusion_df=confusion_df.append(confusion)
#                 try:
#                     confusion_norm=pd.DataFrame(confusion_matrix(y_test,y_predicted,normalize='true'),columns=['0','1'])
#                     confusion_norm['Case']=['0','1']
#                     confusion_norm['Team']=[team,team]
#                 except:
#                     confusion_norm=pd.DataFrame(confusion_matrix(y_test,y_predicted),columns=['1'])
#                     confusion_norm['0']=0
#                     confusion_norm['Case']=['1']
#                     confusion_norm['Team']=[team]
#                 confusion_norm=confusion_norm[['Team','Case','0','1']]
#                 confusion_norm.append([team,'0',0,0])
#                 confusion_norm=confusion_norm[['Team','Case','0','1']]
#                 confusion_norm_df=confusion_norm_df.append(confusion_norm)
        
        MAE_df=pd.DataFrame(MAE_list)
        Accuracy_df=pd.DataFrame(Accuracy_list)
        RMSE_df=pd.DataFrame(RMSE_list)
#         ROC_AUC_df=pd.DataFrame(ROC_AUC_list)
        models_df=pd.DataFrame(models_list)
        team_result=pd.DataFrame(team_order,columns=['Teams'])
        team_result['MAE']=MAE_df
        team_result['Accuracy']=Accuracy_df
        team_result['RMSE']=RMSE_df
#         team_result['ROC_AUC']=ROC_AUC_df
        team_result['Modelo']=models_df
        team_result_grouped=team_result.groupby(['Modelo'],as_index=False).mean()
        best_model=team_result_grouped[(team_result_grouped['Accuracy']==team_result_grouped['Accuracy'].max())]
        
        print(best_model)
        
        season_list=[]
        date_list=[]
        team_list=[]
        against_list=[]
        home_team_list=[]
        away_team_list=[]
        elo_team_list=[]
        elo_rest_team_list=[]
        predicted_result_list=[]
        Accuracy_list=[]
        real_result_list=[]
        for team in DATAFRAME_ENTRENAR['HomeTeam'].unique():
            
            print('Procederemos a predecir los resultados del resto de la temporada para el equipo '+team)
            df_concat_local=DATAFRAME_ENTRENAR[(DATAFRAME_ENTRENAR['HomeTeam']==team)^(DATAFRAME_ENTRENAR['AwayTeam']==team)]
            df_concat_local=df_concat_local[['Season','Date','HomeTeam','AwayTeam','RESULT','GOALSDIFF','TEAM_A','TEAM_B','HOME_TEAM_ELO_BEFORE_GAME','AWAY_TEAM_ELO_BEFORE_GAME']]
            df_concat_local['Ranking_ELO_'+team]=0
            df_concat_local.loc[(df_concat_local.HomeTeam==team),'Ranking_ELO_'+team]=df_concat_local.HOME_TEAM_ELO_BEFORE_GAME
            df_concat_local.loc[(df_concat_local.AwayTeam==team),'Ranking_ELO_'+team]=df_concat_local.AWAY_TEAM_ELO_BEFORE_GAME
            df_concat_local['Ranking_ELO_RESTO_EQUIPOS']=0
            df_concat_local.loc[(df_concat_local.HomeTeam!=team),'Ranking_ELO_RESTO_EQUIPOS']=df_concat_local.HOME_TEAM_ELO_BEFORE_GAME
            df_concat_local.loc[(df_concat_local.AwayTeam!=team),'Ranking_ELO_RESTO_EQUIPOS']=df_concat_local.AWAY_TEAM_ELO_BEFORE_GAME
            df_concat_local['Resultado_'+team]=0
            df_concat_local.loc[(df_concat_local.RESULT=='H')&(df_concat_local.HomeTeam==team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=2
            df_concat_local.loc[(df_concat_local.RESULT=='A')&(df_concat_local.AwayTeam==team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=2
            df_concat_local.loc[(df_concat_local.GOALSDIFF==0),'Resultado_'+team]=1
            df_concat_local.loc[(df_concat_local.RESULT=='H')&(df_concat_local.HomeTeam!=team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=0
            df_concat_local.loc[(df_concat_local.RESULT=='A')&(df_concat_local.AwayTeam!=team)&(df_concat_local.GOALSDIFF!=0),'Resultado_'+team]=0
            df_concat_local_train=df_concat_local[df_concat_local['Season']<'2019-20']
            df_concat_local_test=df_concat_local[df_concat_local['Season']>='2019-20']
            X_train=df_concat_local_train[['Ranking_ELO_RESTO_EQUIPOS','Ranking_ELO_'+team]]
            y_train=df_concat_local_train['Resultado_'+team]
            
            for i in [0,1,2,3,4,5]:
                while True:
                    try:
                        MODEL_TO_USE=(str(best_model['Modelo'][i]))
                    except:
                        pass
                    break
            print(MODEL_TO_USE)
            
            if MODEL_TO_USE == 'LogisticRegression':
                print('A')
                clf= LogisticRegression()
            elif MODEL_TO_USE == 'RandomForestClassifier':
                print('B')
                clf = RandomForestClassifier(max_depth=10, random_state=0)
            elif MODEL_TO_USE == 'LinearDiscriminantAnalysis':
                print('C')
                clf = LinearDiscriminantAnalysis()
            elif MODEL_TO_USE == 'MLPClassifier':
                print('D')
                clf = MLPClassifier()
            elif MODEL_TO_USE == 'DecisionTreeClassifier':
                print('E')
                clf = DecisionTreeClassifier()
            elif MODEL_TO_USE == 'MLPClassifier':
                scaler = StandardScaler()  # doctest: +SKIP
                scaler.fit(X_train)  # doctest: +SKIP
                X_train = scaler.transform(X_train)  
                X_test = scaler.transform(X_test)
            if len(df_concat_local_train.index)==0:
                pass
            else:
                model=clf.fit(X_train, y_train)
            if len(df_concat_local_test.index) == 0:
                    pass
            else:
                print(df_concat_local_test)
                for row in df_concat_local_test.itertuples():
                    team_list.append(team)
                    against_list.append(row.TEAM_B)
                    season=row.Season
                    season_list.append(season)
                    date=row.Date
                    date_list.append(date)
                    home_team=row.HomeTeam
                    home_team_list.append(home_team)
                    away_team=row.AwayTeam
                    away_team_list.append(away_team)
                    elo_team=row[11]
                    elo_team_list.append(elo_team)
                    elo_rest_team=row.Ranking_ELO_RESTO_EQUIPOS
                    elo_rest_team_list.append(elo_rest_team)
                    y_test=np.array([row[-1]])
                    X_test=np.array([[elo_rest_team,elo_team]])
                    y_predicted = model.predict(X_test)
                    predicted_result_list.append(y_predicted)
                    real_result_list.append(y_test)
                    Accuracy_Modelo=accuracy_score(y_test,y_predicted)
                    Accuracy_list.append(Accuracy_Modelo)
                    if y_predicted == 2:
                        print('En el juego '+home_team+' vs ',away_team,' del día ',date,', según el modelo, el equipo ',team,' resultará ganador, cuando realmente resultó en ',y_test)
                    if y_predicted == 1:
                        print('En el juego '+home_team+' vs ',away_team,' del día ',date,',según el modelo, el equipo ',team,' empatará, cuando realmente resultó en ',y_test)
                    if y_predicted == 0:
                        print('En el juego '+home_team+' vs ',away_team,' del día ',date,' según el modelo, el equipo ',team,' resultará perdedor, cuando realmente resultó en ',y_test)
                    print('El modelo tuvo un accuracy del ',Accuracy_Modelo)
        
        date_df=pd.DataFrame(date_list)
        team_df=pd.DataFrame(team_list)
        against_df=pd.DataFrame(against_list)
        home_team_df=pd.DataFrame(home_team_list)
        away_team_df=pd.DataFrame(away_team_list)
        elo_team_df=pd.DataFrame(elo_team_list)
        elo_rest_team_df=pd.DataFrame(elo_rest_team_list)
        predicted_result_df=pd.DataFrame(predicted_result_list)
        real_result_df=pd.DataFrame(real_result_list)
        accuracy_df=pd.DataFrame(Accuracy_list)
        season_df=pd.DataFrame(season_list,columns=['Season'])
        season_df['Date']=date_df
        season_df['Team']=team_df
        season_df['Against']=against_df
        season_df['Home_Team']=home_team_df
        season_df['Away_Team']=away_team_df
        season_df['Elo_Team']=elo_team_df
        season_df['Elo_Rest_Team']=elo_rest_team_df
        season_df['Predicted_Result']=predicted_result_df
        season_df['Real_Result']=real_result_df
        season_df['Accuracy']=accuracy_df
        season_df['Timestamp']=datetime.now()
        
        season_df['Against']=season_df['Against'].replace({0:'Arsenal'})
        season_df['Against']=season_df['Against'].replace({1:'Aston Villa'})
        season_df['Against']=season_df['Against'].replace({2:'Blackburn'})
        season_df['Against']=season_df['Against'].replace({3:'Bolton'})
        season_df['Against']=season_df['Against'].replace({4:'Bournemouth'})
        season_df['Against']=season_df['Against'].replace({5:'Brighton'})
        season_df['Against']=season_df['Against'].replace({6:'Burnley'})
        season_df['Against']=season_df['Against'].replace({7:'Cardiff'})
        season_df['Against']=season_df['Against'].replace({8:'Chelsea'})
        season_df['Against']=season_df['Against'].replace({9:'Crystal Palace'})
        season_df['Against']=season_df['Against'].replace({10:'Everton'})
        season_df['Against']=season_df['Against'].replace({11:'Fulham'})
        season_df['Against']=season_df['Against'].replace({12:'Huddersfield'})
        season_df['Against']=season_df['Against'].replace({13:'Hull'})
        season_df['Against']=season_df['Against'].replace({14:'Leicester'})
        season_df['Against']=season_df['Against'].replace({15:'Liverpool'})
        season_df['Against']=season_df['Against'].replace({16:'Man City'})
        season_df['Against']=season_df['Against'].replace({17:'Man United'})
        season_df['Against']=season_df['Against'].replace({18:'Middlesbrough'})
        season_df['Against']=season_df['Against'].replace({19:'Newcastle'})
        season_df['Against']=season_df['Against'].replace({20:'Norwich'})
        season_df['Against']=season_df['Against'].replace({21:'QPR'})
        season_df['Against']=season_df['Against'].replace({22:'Reading'})
        season_df['Against']=season_df['Against'].replace({23:'Sheffield United'})
        season_df['Against']=season_df['Against'].replace({24:'Southampton'})
        season_df['Against']=season_df['Against'].replace({25:'Stoke'})
        season_df['Against']=season_df['Against'].replace({26:'Sunderland'})
        season_df['Against']=season_df['Against'].replace({27:'Swansea'})
        season_df['Against']=season_df['Against'].replace({28:'Tottenham'})
        season_df['Against']=season_df['Against'].replace({29:'Watford'})
        season_df['Against']=season_df['Against'].replace({30:'West Brom'})
        season_df['Against']=season_df['Against'].replace({31:'West Ham'})
        season_df['Against']=season_df['Against'].replace({32:'Wigan'})
        season_df['Against']=season_df['Against'].replace({33:'Wolves'})
        
        
        season_df.sort_values('Accuracy',ascending=False,inplace=True)
        season_df.drop_duplicates(subset=['Date','Home_Team','Away_Team'],inplace=True)
        season_df.reset_index(drop=True,inplace=True)
        season_df.loc[season_df.Predicted_Result==2,'Winner']=season_df.Team
        season_df.loc[season_df.Predicted_Result==1,'Winner']='Draw'
        season_df.loc[season_df.Predicted_Result==0,'Winner']=season_df.Against
        

        print('---------------------------------------------------------------------------------------') 
        return season_df

def CARGAR_RESULTADO_MIERDA(DATAFRAME_RESULTADO):
    route='My Project 59792-523c1ed8c01a.json'
    credentials = service_account.Credentials.from_service_account_file(route)
    pandas_gbq.context.credentials = credentials
    DATAFRAME_RESULTADO.to_gbq(destination_table='FOOTBALL_PREDICTOR.PREDICTIONS',project_id='long-semiotics-274314',if_exists='append',credentials=credentials)

    return

def IMPORTACION_DATOS_GOOGLE_CLOUD_XG(INDIVIDUAL=False,PARTICULAR=False):
    
    Query=('SELECT*FROM`FOOTBALL_PREDICTOR.PL_HISTORIC` WHERE `Season` <> "2019-20" ORDER BY `Season`,`Date` ASC')
    route='My Project 59792-523c1ed8c01a.json'
    MATCHES=connection_Google_Cloud_Bigquery(route,Query)
    MATCHES=MATCHES.dropna()

    if INDIVIDUAL:
        home=MATCHES[['Season','Date','HomeTeam','FTHG','HTHG','HS','HST','HC','HF','HY','HR']]
        home=home.rename(columns={'HomeTeam':'Team','FTHG':'Goals','HTHG':'HalfTimeGoals','HS':'Shots','HST':'ShotsTarget','HC':'Corners','HF':'Fouls','HY':'Yellows','HR':'Reds'})
        home['local?']='Casa'
        away=MATCHES[['Season','Date','AwayTeam','FTAG','HTAG','AS','AST','AC','AF','AY','AR']]
        away=away.rename(columns={'AwayTeam':'Team','FTAG':'Goals','HTAG':'HalfTimeGoals','AS':'Shots','AST':'ShotsTarget','AC':'Corners','AF':'FoulsCommitted','AY':'Yellows','AR':'Reds'})
        away['local?']='Visitante'
        df_concat=pd.concat([home,away])
        df_concat=df_concat.sort_values('Date')
        df_concat=df_concat.sort_values('Date')
        df_concat=df_concat.reset_index()
        df_concat=df_concat.drop(columns=['index'])
        df_concat['Goal?']=df_concat['Goals']>0
        df_concat['Goal?']=df_concat['Goal?'].replace(True,1)
        df_concat['Goal?']=df_concat['Goal?'].replace(False,0)
        df_concat=df_concat.fillna(0)
    
    return df_concat

def REGRESSION_MODELS(FACTORESTIMAR,DATAFRAME):
    
    MODELS=['GradientBoostingRegressor','MLPRegressor','AdaBoostRegressor','LinearRegression','BaggingRegressor','ExtraTreesRegressor','RandomForestRegressor']
    MAE_list=[]
    RMSE_list=[]
    team_order=[]
    confusion_df=pd.DataFrame()
    confusion_norm_df=pd.DataFrame()
    matriz_confusion_df=pd.DataFrame()
    models_list=[]

    for MODEL in MODELS:
        print('Ejecutaremos el modelo '+MODEL)
        for team in DATAFRAME['Team'].unique():
            print('Procederemos a hacer el modelo '+MODEL+' para '+team)
            df_concat_local=DATAFRAME[DATAFRAME['Team']==team]
            if FACTORESTIMAR == 'shotsXG':
                X=df_concat_local[['ShotsTarget']]
            if FACTORESTIMAR == 'non_shotsXG':
                X=df_concat_local[['Corners', 'Yellows', 'Reds', 'FoulsCommitted']]
            y=df_concat_local[['Goals']]
            if (y.count()<2).bool() == True:
                pass
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                if MODEL == 'GradientBoostingRegressor':
                    print('A')
                    reg=GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,max_depth=1, random_state=0, loss='ls')
                if MODEL == 'MLPRegressor':
                    print('B')
                    reg=MLPRegressor()
                if MODEL == 'AdaBoostRegressor':
                    print('C')
                    reg=AdaBoostRegressor()
                if MODEL == 'LinearRegression':
                    print('D')
                    reg=LinearRegression()
                if MODEL == 'BaggingRegressor':
                    print('E')
                    reg=BaggingRegressor()
                if MODEL == 'ExtraTreesRegressor':
                    print('F')
                    reg=ExtraTreesRegressor()
                if MODEL == 'RandomForestRegressor':
                    print('G')
                    reg=RandomForestRegressor()

                model=reg.fit(X_train,y_train)
                y_predicted = model.predict(X_test)
                y_predicted_team=y_predicted.copy()
                team_order.append(team)
                models_list.append(MODEL)
                MAE=mean_absolute_error(y_test,y_predicted)
                MAE_list.append(MAE)
                RMSE=mean_squared_error(y_test,y_predicted)
                RMSE_list.append(RMSE)
    
    
    MAE_df=pd.DataFrame(MAE_list)
    if MAE_df.empty:
        print('DataFrame is empty!')
        MAE_df=str('NO HAY DATOS SUFICIENTES')
    RMSE_df=pd.DataFrame(RMSE_list)
    if RMSE_df.empty:
        print('DataFrame is empty!')
        RMSE_df=str('NO HAY DATOS SUFICIENTES')
    models_df=pd.DataFrame(models_list)
    if models_df.empty:
        print('DataFrame is empty!')
        models_df=str('NO HAY DATOS SUFICIENTES')
    team_result=pd.DataFrame(team_order,columns=['Teams'])
    team_result['MAE']=MAE_df
    team_result['RMSE']=RMSE_df
    team_result['Modelo']=models_df

    if team_result.empty:
        team_expected_goals_df=['NO HAY DATOS SUFICIENTES']
        pass
    else:
        team_result=team_result[team_result['MAE'] != 'NO HAY DATOS SUFICIENTES']
        print(team_result)
        print(team_result.info())
        model_comparative=team_result.groupby(['Modelo'],as_index=False).mean()
        print(model_comparative)
        best_model=model_comparative[model_comparative['MAE']==model_comparative['MAE'].min()]
        print(best_model)

        #Ahora que tenemos el mejor modelo en base al histórico, creemos una expectativa de goles
        #Para eso, del histórico, tomaremos la media de los últimos 5 partidos jugados 
        #Estos valores serán introducidos en el modelo y se creará un DF con los datos de expectativas de goles

        print(str(best_model['Modelo']))

        expected_goals=[]
        team_expected_goals=[]
        for team in DATAFRAME['Team'].unique():
            print('Procederemos a hacer el modelo '+str(best_model['Modelo'])+' para '+team)
            df_concat_local=DATAFRAME[DATAFRAME['Team']==team]
            valores_introducir=df_concat_local[-5:]
            if FACTORESTIMAR == 'shotsXG':
                X=df_concat_local[['ShotsTarget']]
                X_predict=valores_introducir[['ShotsTarget']]
                X_predict=pd.DataFrame(X_predict.mean()).T
            if FACTORESTIMAR == 'non_shotsXG':
                X=df_concat_local[['Corners', 'Yellows', 'Reds', 'FoulsCommitted']]
                X_predict=valores_introducir[['Corners', 'Yellows', 'Reds', 'FoulsCommitted']]
                X_predict=pd.DataFrame(X_predict.mean()).T
            y=df_concat_local[['Goals']]
            if (y.count()<2).bool() == True:
                pass
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                if str(best_model['Modelo']) == 'GradientBoostingRegressor':
                    print('A')
                    reg=GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,max_depth=1, random_state=0, loss='ls')
                if str(best_model['Modelo']) == 'MLPRegressor':
                    print('B')
                    reg=MLPRegressor()
                if str(best_model['Modelo']) == 'AdaBoostRegressor':
                    print('C')
                    reg=AdaBoostRegressor()
                if str(best_model['Modelo']) == 'LinearRegression':
                    print('D')
                    reg=LinearRegression()
                if str(best_model['Modelo']) == 'BaggingRegressor':
                    print('E')
                    reg=BaggingRegressor()
                if str(best_model['Modelo']) == 'ExtraTreesRegressor':
                    print('F')
                    reg=ExtraTreesRegressor()
                if str(best_model['Modelo']) == 'RandomForestRegressor':
                    print('G')
                    reg=RandomForestRegressor()

                model=reg.fit(X_train,y_train)
                print(X_predict)
                predicted_values=model.predict(X_predict)
                print(type(predicted_values))
                team_expected_goals.append(team)
                expected_goals.append(predicted_values.tolist())
        print(expected_goals)
        print(type(expected_goals))
        expected_goals_df=pd.DataFrame(expected_goals)
        team_expected_goals_df=pd.DataFrame(team_expected_goals,columns=['Teams'])
        if FACTORESTIMAR == 'shotsXG':
            team_expected_goals_df['shotsXG']=expected_goals_df
        elif FACTORESTIMAR == 'non_shotsXG':
            team_expected_goals_df['non_shotsXG']=expected_goals_df
        team_expected_goals_df['timestamp']=datetime.now()
        
        print(team_expected_goals_df)

    return team_expected_goals_df

def CARGAR_RESULTADO_MIERDA_XG_INDIVIDUAL(DATAFRAME_RESULTADO,FACTORESTIMAR):

    route='My Project 59792-523c1ed8c01a.json'
    credentials = service_account.Credentials.from_service_account_file(route)
    pandas_gbq.context.credentials = credentials
    if FACTORESTIMAR == 'shotsXG':
        DATAFRAME_RESULTADO.to_gbq(destination_table='FOOTBALL_PREDICTOR.SHOTS_XG',project_id='long-semiotics-274314',if_exists='replace',credentials=credentials)
    if FACTORESTIMAR == 'non_shotsXG':
        DATAFRAME_RESULTADO.to_gbq(destination_table='FOOTBALL_PREDICTOR.NON_SHOTS_XG',project_id='long-semiotics-274314',if_exists='replace',credentials=credentials)

    return

def LOCURA(DATAFRAME_HISTORICO,DATAFRAME,LOCURA):
    
    if LOCURA == 'shotsXG':
        XG_GENERAL=pd.DataFrame(columns=['Teams','shotsXG','timestamp','model','situacion'])
    else:
        XG_GENERAL=pd.DataFrame(columns=['Teams','non_shotsXG','timestamp','model','situacion'])
    
    for row in DATAFRAME.itertuples():
        
        print('Los Equipos que tengo que comparar son: '+row.HomeTeam+' y '+row.AwayTeam)
        Working_Dataframe=DATAFRAME_HISTORICO[(DATAFRAME_HISTORICO.HomeTeam==row.HomeTeam)&(DATAFRAME_HISTORICO.AwayTeam==row.AwayTeam)]
        
        if LOCURA in ['shotsXG','non_shotsXG']:
            home=Working_Dataframe[['Season','Date','HomeTeam','FTHG','HTHG','HS','HST','HC','HF','HY','HR']]
            home=home.rename(columns={'HomeTeam':'Team','FTHG':'Goals','HTHG':'HalfTimeGoals','HS':'Shots','HST':'ShotsTarget','HC':'Corners','HF':'Fouls','HY':'Yellows','HR':'Reds'})
            home['local?']='Casa'
            away=Working_Dataframe[['Season','Date','AwayTeam','FTAG','HTAG','AS','AST','AC','AF','AY','AR']]
            away=away.rename(columns={'AwayTeam':'Team','FTAG':'Goals','HTAG':'HalfTimeGoals','AS':'Shots','AST':'ShotsTarget','AC':'Corners','AF':'FoulsCommitted','AY':'Yellows','AR':'Reds'})
            away['local?']='Visitante'
            df_concat=pd.concat([home,away])
            df_concat=df_concat.sort_values('Date')
            df_concat=df_concat.sort_values('Date')
            df_concat=df_concat.reset_index()
            df_concat=df_concat.drop(columns=['index'])
            df_concat['Goal?']=df_concat['Goals']>0
            df_concat['Goal?']=df_concat['Goal?'].replace(True,1)
            df_concat['Goal?']=df_concat['Goal?'].replace(False,0)
            df_concat=df_concat.fillna(0)
            print(df_concat)
            XG=REGRESSION_MODELS(LOCURA,df_concat)
            print(str(type(XG)))

            if str(type(XG))== "<class 'list'>":
                pass
            else:
                XG['situacion']=row.HomeTeam+'-'+row.AwayTeam
                XG_GENERAL=pd.concat([XG_GENERAL,XG],ignore_index=True)

            print(XG_GENERAL)
                          
    return XG_GENERAL

def QUERY_PARTICULAR(HISTORICO=False,TEMPORADA=False):

    route='My Project 59792-523c1ed8c01a.json'
    Query_Historic=('SELECT*FROM`FOOTBALL_PREDICTOR.PL_HISTORIC` ORDER BY `Season`,`Date` ASC')
    MATCHES_Historic=connection_Google_Cloud_Bigquery(route,Query_Historic)
    MATCHES_Historic=MATCHES_Historic.dropna()

    if HISTORICO:
        MATCHES_Historic=MATCHES_Historic[MATCHES_Historic.Season!=TEMPORADA]
    else:
        MATCHES_Historic=MATCHES_Historic[MATCHES_Historic.Season==TEMPORADA]

    return     MATCHES_Historic

def CARGAR_XG_JUEGOS(DATAFRAME_RESULTADO,FACTORESTIMAR):

    route='My Project 59792-523c1ed8c01a.json'
    credentials = service_account.Credentials.from_service_account_file(route)
    pandas_gbq.context.credentials = credentials
    if FACTORESTIMAR == 'shotsXG':
        DATAFRAME_RESULTADO.to_gbq(destination_table='FOOTBALL_PREDICTOR.SHOTS_XG_JUEGOS',project_id='long-semiotics-274314',if_exists='replace',credentials=credentials)
    if FACTORESTIMAR == 'non_shotsXG':
        DATAFRAME_RESULTADO.to_gbq(destination_table='FOOTBALL_PREDICTOR.NON_SHOTS_XG_JUEGOS',project_id='long-semiotics-274314',if_exists='replace',credentials=credentials)

    return

def CARGAR_ELO_HISTORICO(HISTORICO_ELO):
    HISTORICO_ELO=HISTORICO_ELO.reset_index()
    HISTORICO_ELO=HISTORICO_ELO.rename(columns={'index':'date',0:'Arsenal',1:'Aston Villa',2:'Blackburn',3:'Bolton',4:'Bournemouth',5:'Brighton',6:'Burnley',7:'Cardiff',8:'Chelsea',9:'Crystal Palace',10:'Everton',11:'Fulham',12:'Huddersfield',13:'Hull',14:'Leicester',15:'Liverpool',16:'Man City',17:'Man United',18:'Middlesbrough',19:'Newcastle',20:'Norwich',21:'QPR',22:'Reading',23:'Sheffield United',24:'Southampton',25:'Stoke',26:'Sunderland',27:'Swansea',28:'Tottenham',29:'Watford',30:'West Brom',31:'West Ham',32:'Wigan',33:'Wolves'})
    ELO_HISTORIC=pd.DataFrame(columns=['date','Team','ELO'])
    for column in HISTORICO_ELO.columns.drop(['date']):
        df=HISTORICO_ELO[['date',column]]
        df['Team']=column
        df=df.rename(columns={column:'ELO'})
        df=df[{'date','Team','ELO'}]
        df=df.dropna()
        ELO_HISTORIC=pd.concat([ELO_HISTORIC,df])
        print(df)
    ELO_HISTORIC
    route='My Project 59792-523c1ed8c01a.json'
    credentials = service_account.Credentials.from_service_account_file(route)
    pandas_gbq.context.credentials = credentials
    ELO_HISTORIC.to_gbq(destination_table='DW_FOOTBALL.FCT_ELO',project_id='long-semiotics-274314',if_exists='replace',credentials=credentials)
