
import time
import datetime

import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

### -------------------------------------- FUNÇÕES RELACIONADAS À PADRONIZAÇÃO MENSAL DE DATAS -------------------------------------- ###

def month_year_to_date(df, year_name = "ano_exercicio", month_name = "mes_referencia"):
    '''
        Converte as informações separadas em colunas de mês e ano em uma única coluna com valores de datas.
    '''
    dates = {
        "year": df[year_name],
        "month": df[month_name],
        "day": np.repeat(15, df.shape[0])
    }
    df = df.copy()
    df = df.assign( data = pd.to_datetime(dates) )
    df = df.filter(["data", "ds_municipio", "ds_orgao", "ds_elemento", "id_despesa", "vl_despesa"])
    return df

def date_to_month_year(df, date_name = "data",):
    '''
        Converte as informações em uma única coluna com valores de datas em colunas de mês e ano.
    '''
    year = df.data.dt.year
    month = df.data.dt.month
    df = df.copy()
    df = df.assign( ano_exercicio = year, mes_referencia = month )
    df = df.filter(["ds_municipio", "ds_orgao", "ds_elemento", "id_despesa", "ano_exercicio", "mes_referencia", "vl_despesa"])
    return df

def agrupar_dataframe(df, list_columns = ["data", "vl_despesa"]):
    '''
        Desagrupa as variáveis do DataFrame, passando linhas individuais para listas.
    '''
    index_variables = ["ds_municipio", "ds_orgao", "id_despesa"]
    
    # Tabela com apenas os índices
    df_grouped = df.groupby(index_variables).size().reset_index().rename(columns = {0:"tam_serie"})
    
    for column in list_columns:
        column_grouped = df.groupby(index_variables)[column].apply(lambda x : np.array(list(x))).reset_index()
        df_grouped = df_grouped.merge(column_grouped, on = index_variables).reset_index(drop = True)
    
    return df_grouped

def desagrupar_dataframe(df_grouped, explode_candidates = ["data", "vl_despesa"]):
    '''
    Desagrupa as variáveis do DataFrame, passando listas para linhas individuais.
    '''
    columns_to_explode = []
    for column in explode_candidates:
        if(column in df_grouped.columns):
            columns_to_explode.append(column)
    
    df = df_grouped.explode(columns_to_explode)
    
    if("ano_exercicio" in df.columns):
        df.ano_exercicio = df.ano_exercicio.astype(int)
    if("vl_despesa" in df.columns):
        df.vl_despesa = df.vl_despesa.astype(np.float64)
    
    df = df.reset_index(drop=True)
    
    return df

def plot_series(df_grouped, j):
    '''
    Constrói o gráfico de uma série de despesas na linha j de um DataFrame já agrupado.
    '''
    t = df_grouped.loc[j,"data"]
    y = df_grouped.loc[j,"vl_despesa"] # valor da despesa
    z = np.log( y[1:]/y[:-1] ) # log-retornos

    fig, ax = plt.subplots(figsize = (16, 6), nrows = 1, ncols = 2)

    ax[0].plot(t, y)
    ax[0].set_title("{} - {}".format(df_grouped.ds_orgao[j], df_grouped.id_despesa[j]))
    
    ax[1].plot(t[1:], z)
    ax[1].set_title("Log-retornos {} - {}".format(df_grouped.ds_orgao[j], df_grouped.id_despesa[0]))
        
    plt.show()
    
### -------------------------------------- FUNÇÕES RELACIONADAS ÀO AGRUPAMENTO BIMESTRAL/QUADRIMESTRAL/SEMESTRAL DE DATAS -------------------------------------- ###

def bimestral_agg(df):
    '''
        Agrega os dados segundo bimestres.
    '''
    df_aux = df.copy()
    
    bim = df_aux.data.swifter.progress_bar(False).apply(lambda x : 1 if(x.month <= 2) else 2 if(x.month <= 4) else 3 if(x.month <= 6) else 4 if(x.month <= 8) else 5 if(x.month <= 10) else 6)
    years = df_aux.data.swifter.progress_bar(False).apply(lambda x : x.year)
    
    df_aux = df_aux.assign(ano = years)
    df_aux = df_aux.assign(bimestre = bim)
    
    df_aux = df_aux.groupby(["ds_municipio", "ds_orgao", "id_despesa", "ano", "bimestre"]).agg({"vl_despesa": "sum"}).reset_index()
    
    # Converte as informações de ano e bimestre para o dia 15 do primeiro mês do bimestre correspondente
    date = df_aux.loc[:,["ano", "bimestre"]].swifter.progress_bar(False).apply(lambda x : datetime.date(int(x[0]), 2*int(x[1])-1, 15), axis = 1)
    df_aux = df_aux.assign(data = date)
    
    df_aux = df_aux.filter(["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"])
    df_aux.data = pd.to_datetime(df_aux.data)
    
    return df_aux

def quadrimestral_agg(df):
    '''
        Agrega os dados segundo quadrimestres.
    '''
    df_aux = df.copy()
    
    quad = df_aux.data.swifter.progress_bar(False).apply(lambda x : 1 if(x.month <= 4) else 2 if(x.month <= 8) else 3)
    years = df_aux.data.swifter.progress_bar(False).apply(lambda x : x.year)
    
    df_aux = df_aux.assign(ano = years)
    df_aux = df_aux.assign(quadrimestre = quad)

    df_aux = df_aux.groupby(["ds_municipio", "ds_orgao", "id_despesa", "ano", "quadrimestre"]).agg({"vl_despesa": "sum"}).reset_index()
    
    # Converte as informações de ano e quadrimestre para o dia 15 do primeiro mês do quadrimestre correspondente
    date = df_aux.loc[:,["ano", "quadrimestre"]].swifter.progress_bar(False).apply(lambda x : datetime.date(int(x[0]), 4*int(x[1])-3, 15), axis = 1)
    df_aux = df_aux.assign(data = date)
    
    df_aux = df_aux.filter(["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"])
    df_aux.data = pd.to_datetime(df_aux.data)
    
    return df_aux

def semestral_agg(df):
    '''
        Agrega os dados segundo semestres.
    '''
    df_aux = df.copy()
    
    sem = df_aux.data.swifter.progress_bar(False).apply(lambda x : 1 if(x.month <= 6) else 2)
    years = df_aux.data.swifter.progress_bar(False).apply(lambda x : x.year)
    
    df_aux = df_aux.assign(ano = years)
    df_aux = df_aux.assign(semestre = sem)
    
    df_aux = df_aux.groupby(["ds_municipio", "ds_orgao", "id_despesa", "ano", "semestre"]).agg({"vl_despesa": "sum"}).reset_index()
    
    # Converte as informações de ano e semestre para o dia 15 do primeiro mês do semestre correspondente
    date = df_aux.loc[:,["ano", "semestre"]].swifter.progress_bar(False).apply(lambda x : datetime.date(int(x[0]), 6*int(x[1])-5, 15), axis = 1)
    df_aux = df_aux.assign(data = date)
    
    df_aux = df_aux.filter(["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"])
    df_aux.data = pd.to_datetime(df_aux.data)
    
    return df_aux

def date_to_bimestral(df, date_name = "data"):
    '''
        Recebe dados agrupados pela função bimestral_agg e converte as datas para colunas separadas de ano e bimestre (1 a 6).
    '''
    year = df[date_name].dt.year
    month = df[date_name].dt.month
    
    bimester = month.swifter.progress_bar(False).apply(lambda x : 1 if(x < 3 ) else 2 if(x < 5) else 3 if(x < 7) else 4 if(x < 9) else 5 if(x < 11) else 6).reset_index(drop = True)
    df = df.copy()
    df = df.assign(ano_exercicio = year, bimestre = bimester)
    df = df.filter(["ds_municipio", "ds_orgao", "id_despesa", "ano_exercicio", "bimestre", "vl_despesa"])
    
    return df

def date_to_quadrimestral(df, date_name = "data"):
    '''
        Recebe dados agrupados pela função quadrimestral_agg e converte as datas para colunas separadas de ano e quadrimestre (1 a 3).
    '''
    year = df[date_name].dt.year
    month = df[date_name].dt.month
    
    quadrimester = month.swifter.progress_bar(False).apply(lambda x : 1 if(x < 5) else 2 if(x < 9) else 3)
    df = df.copy()
    df = df.assign(ano_exercicio = year, quadrimestre = quadrimester)
    df = df.filter(["ds_municipio", "ds_orgao", "id_despesa", "ano_exercicio", "quadrimestre", "vl_despesa"])
    
    return df

def date_to_semester(df, date_name = "data"):
    '''
        Recebe dados agrupados pela função semestral_agg e converte as datas para colunas separadas de ano e semestre (1 a 2).
    '''
    year = df[date_name].dt.year
    month = df[date_name].dt.month
    
    semester = month.swifter.progress_bar(False).apply(lambda x : 1 if(x < 7) else 2).reset_index(drop = True)
    df = df.copy()
    df = df.assign(ano_exercicio = year, semestre = semester)
    df = df.filter(["ds_municipio", "ds_orgao", "id_despesa", "ano_exercicio", "semestre", "vl_despesa"])
    
    return df

def bimester_to_date(df, filtered_columns = ["data", "ds_municipio", "ds_orgao", "ds_elemento", "id_despesa", "vl_despesa"]):
    '''
        Converte a configuração em ano/bimestre em datas
    '''
    dates = {
        "year": df.ano_exercicio,
        "month": df.bimestre.swifter.progress_bar(False).apply(lambda x : 1 if(x == 1) else 3 if(x == 2) else 5 if(x == 3) else 7 if(x == 4) else 9 if(x == 5) else 11),
        "day": np.repeat(15, df.shape[0])
    }
    df = df.copy()
    df = df.assign( data = pd.to_datetime(dates) )
    df = df.filter(filtered_columns)
    return df

def quadrimester_to_date(df, filtered_columns = ["data", "ds_municipio", "ds_orgao", "ds_elemento", "id_despesa", "vl_despesa"]):
    '''
        Converte a configuração em ano/quadrimestre em datas
    '''
    dates = {
        "year": df.ano_exercicio,
        "month": df.quadrimestre.swifter.progress_bar(False).apply(lambda x : 1 if(x == 1) else 5 if(x == 2) else 9),
        "day": np.repeat(15, df.shape[0])
    }
    df = df.copy()
    df = df.assign( data = pd.to_datetime(dates) )
    df = df.filter(filtered_columns)
    return df

def semester_to_date(df, filtered_columns = ["data", "ds_municipio", "ds_orgao", "ds_elemento", "id_despesa", "vl_despesa"]):
    '''
        Converte a configuração em ano/semestre em datas
    '''
    dates = {
        "year": df.ano_exercicio,
        "month": df.semestre.swifter.progress_bar(False).apply(lambda x : 1 if(x == 1) else 7),
        "day": np.repeat(15, df.shape[0])
    }
    df = df.copy()
    df = df.assign( data = pd.to_datetime(dates) )
    df = df.filter(filtered_columns)
    return df

### -------------------------------------- FUNÇÕES RELACIONADAS À PADRONIZAÇÃO DAS SÉRIES DE DESPESAS -------------------------------------- ###

class Padronizacao:
    
    def fit_despesa(self, df_grouped):
        
        mean_despesa = df_grouped.vl_despesa.swifter.progress_bar(False).apply(lambda x : np.mean(x))
        sd_despesa = df_grouped.vl_despesa.swifter.progress_bar(False).apply(lambda x : np.std(x))
        
        # Dados individuais de cada série de despesas
        meansd_info = df_grouped.loc[:,["ds_municipio", "ds_orgao", "id_despesa"]]
        meansd_info["mean_despesa"] = mean_despesa
        meansd_info["sd_despesa"] = sd_despesa

        self.meansd_info_despesa = meansd_info
    
    def transform_despesa(self, df_grouped, keep_all = False):
        # Cria uma tabela auxiliar apenas com as informações úteis para o método
        df_grouped_aux = (
            df_grouped.loc[:,["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"]].
            merge(self.meansd_info_despesa, on = ["ds_municipio", "ds_orgao", "id_despesa"]).
            reset_index(drop = True)
        )
        
        # Aplica a função de padronização para cada linha da tabela
        df_grouped_aux["vl_despesa"] = df_grouped_aux.loc[:,["vl_despesa", "mean_despesa", "sd_despesa"]].swifter.progress_bar(False).apply(
            lambda x : (np.array(x[0]) - x[1]) / x[2] if x[2] != 0 else None,
            axis = 1
        )
        
        # Se deseja-se manter todas as observações, inclusive as que não foram transformadas por falta de informação
        if(keep_all):
            return df_grouped_aux.loc[:,["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"]]
            
        # Se os valores de mínimo e máximo da série de uma determinada despesa forem idênticos, exibe um aviso
        if(df_grouped_aux.vl_despesa.isna().any()):
            print("*** Séries com informações insuficientes removidas automaticamente ***")
            
        return df_grouped_aux.loc[df_grouped_aux["vl_despesa"].notna(),["ds_municipio", "ds_orgao", "id_despesa", "data", "vl_despesa"]]
    
    def rescale_despesa(self, df_grouped_scaled, rescale_variables = ["vl_despesa"]):
        df_grouped_scaled_aux = (
            df_grouped_scaled.loc[:,["ds_municipio", "ds_orgao", "id_despesa", "data"] + rescale_variables].
            merge(
                self.meansd_info_despesa, on = ["ds_municipio", "ds_orgao", "id_despesa"]
            )
        )
        
        for v in rescale_variables:
            df_grouped_scaled_aux[ v ] = df_grouped_scaled_aux.loc[:,[v, "mean_despesa", "sd_despesa"]].swifter.progress_bar(False).apply(
                lambda x : x[0] * x[2] + x[1],
                axis = 1
        )
        
        return df_grouped_scaled_aux.loc[:,["ds_municipio", "ds_orgao", "id_despesa", "data"] + rescale_variables]
    
### -------------------------------------- FUNÇÕES RELACIONADAS À CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS -------------------------------------- ###

def label_encoder(df, categorical_columns = ["ds_municipio", "ds_orgao", "id_despesa"]):
    df = df.copy()
    
    le = LabelEncoder()
    le_dict = {}

    for column in categorical_columns:
        df.loc[:,column] = le.fit_transform( df.loc[:,column] )
        le_dict[column] = dict(zip( np.arange( len(le.classes_) ), le.classes_) )
    
    return df, le_dict

def label_decoder(df, encoder_dict, categorical_columns = ["ds_municipio", "ds_orgao", "id_despesa"]):
    df = df.copy()

    for column in categorical_columns:
        df.loc[:, column] = [ encoder_dict[column][code] for code in df.loc[:, column] ]
        
    return df