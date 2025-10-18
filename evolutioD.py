import numpy as np
import random
from codific import decodificar_vector_de, codificar_configuracion_de
from evaluate_config import valuation, valuationI

def evolution_differential(X, y, population_size=5, dimension=7, F=0.8, CR=0.9, num_generations=10):
    normalModel = valuationI(X, y)

    cambiol = 0
    cambiog = 0
    population = np.random.rand(population_size, dimension)
    aptitud = np.zeros(population_size)

    # Evaluación inicial
    for i in range(population_size):
        config = decodificar_vector_de(population[i])
        try:
            aptitud[i] = valuation(config, X, y)
        except Exception:
            aptitud[i] = -np.inf

    best_apti = np.max(aptitud)
    best_gen = population[np.argmax(aptitud)]

    # Evolución
    for gen in range(num_generations):
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            r1, r2, r3 = random.sample(indices, 3)

            # Mutación
            mutant_vector = population[r1] + F * (population[r2] - population[r3])
            mutant_vector = np.clip(mutant_vector, 0, 1)

            # Cruce (binomial)
            test_vector = np.copy(population[i])
            j_rand = random.randint(0, dimension - 1)
            for j in range(dimension):
                if random.random() <= CR or j == j_rand:
                    test_vector[j] = mutant_vector[j]

            # Selección
            test_config = decodificar_vector_de(test_vector)
            try:
                test_apti = valuation(test_config, X, y)
            except Exception:
                test_apti = -np.inf

            if test_apti > aptitud[i]:
                population[i] = test_vector
                aptitud[i] = test_apti
                cambiol += 1

                if test_apti > best_apti:
                    best_apti = test_apti
                    best_gen = test_vector
                    cambiog += 1

        print(f"Generación {gen+1:02d} | Mejor aptitud: {best_apti:.4f}")

    mejor_confi = decodificar_vector_de(best_gen)
    print(f"\nCambios locales: {cambiol}")
    print(f"Cambios globales: {cambiog}")
    print(f"Score base (sin optimización): {normalModel:.4f}")
    print("\nMejor configuración encontrada:")
    print(f"Aptitud: {best_apti:.4f}")
    print(f"Configuración: {mejor_confi}")

    return mejor_confi, best_apti, best_gen
