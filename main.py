import pandas as pd
import numpy as np
from evolutioD import evolition_differential


data= pd.read_csv('./data/data.tsv', sep='\t')
X = data['tweet']
y = data['offensive']

mejor_config_de = evolition_differential(X, y, population_size=5,num_gene=30)
print("\nConfiguración óptima encontrada por DE:", mejor_config_de)