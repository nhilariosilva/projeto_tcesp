import numpy as np
import requests

def consulta_ipca(meses, anos, d = 2):
    '''
        Obtém os números-índice dos períodos especificados.
        args:
            meses: int/list
                Meses a serem consultados
            anos: int/list
                Anos a serem consultados (deve ter o mesmo tamanho de ``meses``)
            d: int
                Número de casas decimais a serem retornadas pela API
    '''
    if(type(meses) != list and type(meses) != np.ndarray): meses = [meses]
    if(type(anos) != list and type(anos) != np.ndarray): anos = [anos]
        
    # Criação da URL
    t = 1737
    v = 2266
    ps = []
    for mes, ano in zip(meses, anos):
        ps.append( "{}{}".format(str(ano), str(mes).zfill(2)) )
    p = ",".join(ps)
    url = "https://apisidra.ibge.gov.br/values/t/{}/n1/all/v/{}/p/{}/d/{}".format(t, v, p, d)
    
    # Requisição à API
    r = requests.get(url).json()
    vs = list(map(lambda x : float(x["V"]), r[1:]))
    
    if(len(meses) == 1): return vs[0]
    
    return vs

def deflacionar_precos(values, meses, anos, mes_ref, ano_ref, d = 2):
    '''
        Converte os preços dos períodos listados para o preço deflacionado no período de referência.
        args:
            values: list
                Preços ao longo dos períodos especificados
            meses: list
                Meses a serem deflacionados
            anos: list
                Meses a serem deflacionados
            mes_ref: int
                Mês de referência para os valores deflacionados
            ano_ref: int
                Ano de referência para os valores deflacionados
            d: int
                Número de casas decimais a serem retornadas pela API
    '''
    # Consulta na API
    i_values = consulta_ipca(meses, anos, d = d)
    i_ref = consulta_ipca(mes_ref, ano_ref, d = d)
    
    if(type(values) != list): values = [values]
    if(type(i_values) != list): i_values = [i_values]
    
    # Deflaciona os preços
    new_values = []
    for value, i in zip(values, i_values):
        new_values.append( round(value * i_ref/i, 2) )
    
    if(len(values) == 1): return new_values[0]
    
    return new_values