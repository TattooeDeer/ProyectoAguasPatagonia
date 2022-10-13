import pandas as pd
from datetime import datetime
import numpy as np
from glob import glob
import re
from sklearn.preprocessing import StandardScaler
from copy import copy


def full_preproc_exgenas(pr18, df_precipitaciones, df_temperaturas,
                                     attr_list = ['PH', 'Fe', 'TURBIEDAD', 'CT', 'E-COLI', 'COLOR VERD', 'precipitacion', 'temperatura']):
    
    """
    DOCSTRING:
        Esta funcion recibe tres objetos tipo pandas.DataFrame, une la informacion de estos dataframes y realiza un pivoteo de la tabla junto con una agregacion a nivel semanal de los valores
        de cada parametro medido indicado en la lista attr_list.
        
        Retorna un dataframe on la informacion concatenada y agregada a nivel de semana, sin considerar localidad.
        
        
        Params:
            - pr18 (pandas.DataFrame): DataFrame con la informacion de PR18.
            - df_precipitaciones (pandas.DataFrame): dri
            - df_temperaturas (pandas.DataFrame):
            - attr_list (list of <str>): Lista con las columnas a conservar del preprocesamiento, el resto de las columnas seran ignoradas.
        
        Return (pandas.DataFrame):
            - Retorna la información agregada a nivel de semana con la informacion de PR18, temperatura y precipitaciones.
            
    """

    def preproc_data_noHistory(data, target = 'valor', date_col = 'fecha_muestra', attr_col = 'codigo_parametro',
                        user_classes = ['Fe', 'COLOR VERD', 'CT', 'PH', 'E-COLI', 'TURBIEDAD'], 
                        drop_NaN_classes = True, date_freq = 'D'):
        df = copy(data) 
        df.dropna(subset = target, inplace = True)
        
        # columna entregada en date_col debe seguir formato yyyy-mm-dd
        df.fecha_muestra = pd.to_datetime(df[date_col], format = '%Y-%m-%d')
        
        grouped = df.groupby([
            attr_col, pd.Grouper(key=date_col, freq=date_freq)])[target].mean().reset_index().sort_values(by = date_col)
    
        grouped_pivoted = pd.pivot_table(data = grouped, index = date_col, columns = attr_col, values = target)[user_classes]
        grouped_pivoted.columns.name =  None
    
        if drop_NaN_classes: # Para obtener una matriz de atributos siempre densa
            grouped_pivoted.dropna(inplace = True)
        
        return grouped_pivoted
    
    ############## Datos Precipitaciones y Temperaturas (CR2) ##############
    
    df_temperaturas['fecha_muestra'] = pd.to_datetime(df_temperaturas['fecha_muestra']) - pd.to_timedelta(7, unit='d')
    df_precipitaciones['fecha_muestra'] = pd.to_datetime(df_precipitaciones['fecha_muestra']) - pd.to_timedelta(7, unit='d')
    
    # Agrupacion semanal de precipitaciones y temperaturas
    weekly_precipitaciones = df_precipitaciones.groupby([pd.Grouper(key='fecha_muestra', freq='W-MON')]).mean().reset_index()
    weekly_precipitaciones = weekly_precipitaciones.groupby(['fecha_muestra']).mean()
    
    weekly_temperaturas = df_temperaturas.groupby([pd.Grouper(key='fecha_muestra', freq='W-MON')]).mean().reset_index()
    weekly_temperaturas = weekly_temperaturas.groupby(['fecha_muestra']).mean()
    
    # ForwardFilling
    weekly_precipitaciones_noLocalidad = weekly_precipitaciones.reset_index()\
                                .groupby(by = 'fecha_muestra').mean().resample('W-MON').ffill()
    weekly_temperaturas_noLocalidad = weekly_temperaturas.reset_index()\
                                .groupby(by = 'fecha_muestra').mean().resample('W-MON').ffill()
    
    
    ############## PR18 ##############
    # Preproc general para pr18
    pr18_processed = preproc_data_noHistory(pr18, drop_NaN_classes=False, date_freq = 'W-MON')
    
    # Join de datos de temperatura y precip con conjunto de datos de RP18 + forward filling para las fechas que no calcen
    data_exogena_padded = pr18_processed.join(weekly_precipitaciones_noLocalidad).join(weekly_temperaturas_noLocalidad)[attr_list]#\
                                                                                                        
    for col_name, col in data_exogena_padded.iteritems():
        data_exogena_padded[col_name] = data_exogena_padded[col_name].fillna(method = 'ffill')
    
    return data_exogena_padded


def getDataFrame_from_CR2Source(file_path, variable_name):
    localidad_re = re.compile(r'/(\w+)_\w+_\w+\.xlsx')
    codigoLocalidad_dict = {'Coyhaique': 82, 'PuertoChacabuco': 255, 'PuertoCisnes': 256, 'Balmaceda': 22, 'PuertoAysen': 254, 'Cochrane': 62,
    'PuertoIbanez': 257, 'ChileChico': 55}
    
    nombre_localidad = localidad_re.findall(file_path)[0]
    
    precip_data = pd.read_excel(file_path)
    precip_data['fecha_muestra'] = pd.to_datetime(list(map(lambda row: f"{int(row[1]['agno'])}/{int(row[1]['mes'])}/{int(row[1]['dia'])}",
                                                           precip_data.iterrows())))
    
    # insertar una columna constante con el codigo de la localidad para despues hacer el join
    precip_data['codigoLocalidad'] = codigoLocalidad_dict[nombre_localidad] 
    precip_data.rename({'valor': variable_name}, axis = 1, inplace = True)
    
    return precip_data[['fecha_muestra', 'codigoLocalidad', variable_name]]

def buildLaggedFeatures(s,lag=2,dropna=True):
        '''
        Builds a new DataFrame to facilitate regressing over all possible lagged features
        '''
        if type(s) is pd.DataFrame:
            new_dict={}
            for col_name in s:
                new_dict[col_name]=s[col_name]
                # create lagged Series
                for l in range(1,lag+1):
                    new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
            res=pd.DataFrame(new_dict,index=s.index)
        
        elif type(s) is pd.Series:
            the_range=range(lag+1)
            res=pd.concat([s.shift(i) for i in the_range],axis=1)
            res.columns=['lag_%d' %i for i in the_range]
        else:
            print('Only works for DataFrame or Series')
            return None
        if dropna:
            return res.dropna()
        else:
            return res 

def generate_predict_format(df, target_col, attr_list = ['Fe', 'COLOR VERD', 'CT', 'PH', 'E-COLI', 'TURBIEDAD'], lag = 3, transform_method = None, ):
    
    if transform_method != None:
        return transform_method.transform(buildLaggedFeatures(df[attr_list], lag = lag).to_numpy())
    else:
        return buildLaggedFeatures(df[attr_list], lag = lag)
    
    
def generate_train_test(data, target_col, test_size = 12, lag = 3, attr_list = ['Fe', 'COLOR VERD', 'CT', 'PH', 'E-COLI', 'TURBIEDAD']):
    # Construccion de las matrices de entrenamiento y pruebas
    X = buildLaggedFeatures(data[attr_list], lag=lag).drop(columns = attr_list).iloc[test_size-1:,:]
    Y = buildLaggedFeatures(data[target_col].iloc[lag:], lag=test_size-1)
    
    std_exogen = StandardScaler()
    
    x_train = X.iloc[:-test_size, :].to_numpy()
    x_test = X.iloc[-test_size:, :].to_numpy()
    
    x_train = std_exogen.fit_transform(x_train)
    x_test = std_exogen.transform(x_test)
    
    y_train = Y.iloc[:-test_size]
    y_test = Y.iloc[-test_size:]
    
    return {'data': (x_train, x_test, y_train, y_test), 'std_scaler': std_exogen}
    