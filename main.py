import pandas as pd
import numpy as np
from evolutioD import evolution_differential


data= pd.read_csv('./data/data.tsv', sep='\t')
X = data['tweet']
y = data['offensive']

mejor_config_de = evolution_differential(X, y, population_size=5)
print("\nConfiguración óptima encontrada por DE:", mejor_config_de)