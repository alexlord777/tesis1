import numpy as np
import random
import time
import csv, json, os
from codific import decodificar_vector_de, codificar_configuracion_de
from evaluate_config import valuation, valuationI


# Función auxiliar: registrar resultados en CSV
def agregar_fila(generacion: int, individuo: int, score: float, config: dict, tiempo: float, filename="resultados.csv"):
    # Crear archivo si no existe
    if not os.path.isfile(filename):
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["generacion", "individuo", "score", "config", "tiempo_ejecucion"])
            writer.writeheader()

    # Agregar fila
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["generacion", "individuo", "score", "config", "tiempo_ejecucion"])
        writer.writerow({
            "generacion": generacion,
            "individuo": individuo,
            "score": score,
            "config": json.dumps(config),
            "tiempo_ejecucion": tiempo
        })



# Algoritmo de Evolución Diferencial

def evolution_differential(X, y, population_size=5, dimension=7, F=0.8, CR=0.9, num_generations=10):
    normalModel = valuationI(X, y)

    cambiol = 0
    cambiog = 0
    population = np.random.rand(population_size, dimension)
    aptitud = np.zeros(population_size)
    print("here")
    # Evaluación inicial
   
    for i in range(population_size):
        config = decodificar_vector_de(population[i])
        try:
            start_time = time.time()
            aptitud[i] = valuation(config, X, y)
            elapsed_time = time.time() - start_time
        except Exception:
            aptitud[i] = -np.inf
            elapsed_time = 0.0

        # Registrar en CSV
        agregar_fila(0, i, aptitud[i], config, elapsed_time)

    best_apti = np.max(aptitud)
    best_gen = population[np.argmax(aptitud)]


    # Proceso evolutivo

    for gen in range(1, num_generations + 1):
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            r1, r2, r3 = random.sample(indices, 3)

            # Mutación
            mutant_vector = population[r1] + F * (population[r2] - population[r3])
            mutant_vector = np.clip(mutant_vector, 0, 1)

            # Cruce
            test_vector = np.copy(population[i])
            j_rand = random.randint(0, dimension - 1)
            for j in range(dimension):
                if random.random() <= CR or j == j_rand:
                    test_vector[j] = mutant_vector[j]

            # Evaluar nuevo individuo con tiempo
            test_config = decodificar_vector_de(test_vector)
            try:
                start_time = time.time()
                test_apti = valuation(test_config, X, y)
                elapsed_time = time.time() - start_time
            except Exception:
                test_apti = -np.inf
                elapsed_time = 0.0

            # Registrar resultado
            agregar_fila(gen, i, test_apti, test_config, elapsed_time)

            # Selección
            if test_apti > aptitud[i]:
                population[i] = test_vector
                aptitud[i] = test_apti
                cambiol += 1

                if test_apti > best_apti:
                    best_apti = test_apti
                    best_gen = test_vector
                    cambiog += 1

        print(f"Generación {gen:02d} | Mejor aptitud: {best_apti:.4f}")


    # Resultados finales
    mejor_confi = decodificar_vector_de(best_gen)
    print(f"\nCambios locales: {cambiol}")
    print(f"Cambios globales: {cambiog}")
    print(f"Score base (sin optimización): {normalModel:.4f}")
    print("\nMejor configuración encontrada:")
    print(f"Aptitud: {best_apti:.4f}")
    print(f"Configuración: {mejor_confi}")

    print("\nArchivo 'resultados.csv' actualizado con tiempos de ejecución.")
    return mejor_confi, best_apti, best_gen
