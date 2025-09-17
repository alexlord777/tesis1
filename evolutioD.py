import numpy as np 
from codific import decodificar_vector_de,codificar_configuracion_de
from evaluate_config import valuation,valuationI
import random

def evolition_diferential(X,y,population_size=5, num_gene=2, F=0.8, CR=0.9):

    normalModel=valuationI(X,y)

    cambiol=0
    cambiog=0
    dimention=7
    population=np.random.rand(population_size,dimention)
    aptitud=np.zeros(population_size)

    for i in range(population_size):
        config=decodificar_vector_de(population[i])
        #print(config)
        aptitud[i]=valuation(config,X,y)

    best_apti=np.max(aptitud)
    best_gen=population[np.argmax(aptitud)]


    for num in range(2):
        for i in range(population_size):
            indiex=[idx for idx in range(population_size) if idx !=i]
            r1,r2,r3= random.sample(indiex,3)

            #Mutation
            mutant_vector=population[r1] + F*(population[r2]-population[r3])
            mutant_vector=np.clip(mutant_vector,0,1)

            #crossing
            test_vector=np.copy(population[i])
            j_rand=random.randint(0,dimention-1)
            for j in range(dimention):
                random.random() <= CR or j==j_rand
                test_vector[j]= mutant_vector[j]
            #Selection 
            test_configuration=decodificar_vector_de(test_vector)
            test_apti=valuation(test_configuration,X,y)

            if test_apti > aptitud[i]:
                population[i]=test_vector
                aptitud[i]=test_apti
                cambiol+=1

                if test_apti>best_apti:
                    cambiog+=1
                    best_apti=test_apti
                    best_gen=test_vector


            
        print(f"Geeracion {num+1}: Mejor aptitud {best_apti}: Mejor configuracion {best_gen}")

    mejor_confi=decodificar_vector_de(best_gen)
    print(cambiol)
    print(cambiog)
    print(f"Score sin configuracion: {normalModel}")
    print("\nMejor configuración encontrada1:")
    print(f"Aptitud: {best_apti:.4f}")
    print(f"Configuración: {mejor_confi}")

    return mejor_confi