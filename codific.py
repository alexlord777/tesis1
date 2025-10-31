import numpy as np

espacio_parametros = {
    'lc': [False, True],
    'del_dup': [False, True],
    'token_list': [[-1], [3], [(2, 1)], [-1, 3]],
    'weighting': ['tf', 'tfidf'],
    'del_diac': [False, True],
    'select_conn': [False, True],
    'token_min_filter': list(np.linspace(0.0, 1.0, 10))  
}


# Mapas para convertir entre hiperparámetros y números
mapeo_token_list = {tuple([-1]): 0.0, tuple([3]): 0.25, tuple([(2, 1)]): 0.5, tuple([-1, 3]): 0.75}
decodificacion_token_list_de = {v: k for k, v in mapeo_token_list.items()}

mapeo_weighting = {'tf': 0.0, 'tfidf': 1.0}
decodificacion_weighting_de = {v: k for k, v in mapeo_weighting.items()}

def codificar_configuracion_de(config):
    vector = np.zeros(7)
    vector[0] = 1.0 if config['lc'] else 0.0
    vector[1] = 1.0 if config['del_dup'] else 0.0
    vector[2] = mapeo_token_list[tuple(config['token_list'])]
    vector[3] = mapeo_weighting[config['weighting']]
    vector[4] = 1.0 if config['del_diac'] else 0.0
    vector[5] = 1.0 if config['select_conn'] else 0.0
    vector[6] = config['token_min_filter']
    return vector

def dec_list(n):
    if n <= 0.125: return [-1]
    elif n <= 0.375: return [3]
    elif n <= 0.625: return [2, 1]
    else: return [-1, 3]

def dec_wei(n):
    return 'tf' if n <= 0.5 else 'tfidf'

def decodificar_vector_de(vector):
    config = {}
    config['lc'] = True if vector[0] >= 0.5 else False
    config['del_dup'] = True if vector[1] >= 0.5 else False
    config['token_list'] = dec_list(vector[2])
    config['weighting'] = dec_wei(vector[3])
    config['del_diac'] = True if vector[4] >= 0.5 else False
    config['select_conn'] = True if vector[5] >= 0.5 else False
    # redondeamos al valor más cercano entre los 10 posibles
    posibles_min_filter = np.linspace(0.0, 1.0, 10)
    config['token_min_filter'] = min(posibles_min_filter, key=lambda x: abs(x - vector[6]))
    return config
