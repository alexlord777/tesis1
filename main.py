import pandas as pd
from evolutioD import evolution_differential
import time


start_time = time.time()


data= pd.read_csv('./data/data.tsv', sep='\t')
#print(data.head())
X = data['tweet']
y = data['offensive']

mejor_config_de = evolution_differential(X, y, population_size=5)

total_time = time.time() - start_time
print(f"Tiempo total de ejecución: {total_time/60:.2f} minutos")