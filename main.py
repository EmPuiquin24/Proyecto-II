import numpy as np
from sklearn.datasets import load_digits
import cv2
import matplotlib.pyplot as plt

# --------------------------------- PARTE A----------------------------------------

# ---------- EN TODO ESTE BLOQUE, SE CARGA EL "DICCIONARIO" DIGITS Y SUS KEYS: "IMAGES" Y "TARGET" ----------

# Se carga el "diccionario de digits en el data set"
digits = load_digits()

# Se carga las keys de images (Lista de matrices) y target (lista de los números del dataset)
imagenes = digits.images
etiquetas = digits.target


# ---------- EN TODO ESTE BLOQUE, SE BUSCA MEDIANTE EL FOR, AÑADIR POR ORDEN (0 al 10) LAS MATRICES Y PROMEDIARLAS Y ESCALARLAS ----------  

matrices_promedio_redimensionadas = [] # SE INICIALIZA LA LISTA QUE CONTENDRÁ A LAS MATRICES PROMEDIO REDIMENSIONADAS

for digito in range(10):
    imagenes_digito = imagenes[etiquetas == digito] # Solo se seleccionará en caso la etiquteta sea igual al dígito en el que se está iterando.
    # Nota personal: Esto tiene sentido por un concepto llamado "Máscara Booleana", la verdad no investigué más sobre esto pero
    # Se puede hacer en arrays (en este caso "etiquetas" es el array), desconozco si se puede hacer en listas u otras aplicaciones.

    matriz_promedio = np.mean(imagenes_digito, axis=0) # Se ponderiará en base a las filas
    # Explicación: Se toma la fila de cada de matriz dentro imagenes_digito, y luego se "itera" y se promedia todo en base a una misma columna
    # Al final, el return será una matriz donde la primera fila representa ese promedio de las primeras filas, la segunda ...
    
    matriz_promedio_redimensionada = cv2.resize(matriz_promedio, (8, 8)).astype(int) # Se redimensiona esa promedio a 8 filas y 8 columnas
    #Esto no requiere más explicación

    matrices_promedio_redimensionadas.append(matriz_promedio_redimensionada) # Se agrega a "matrices_promedio_redimensionadas"
    # Esto mucho menos :v


print()
print()
digito_seleccionado = int(input("Ingrese el dígito del cual desea ver la matriz promedio (0-9): "))
print("-"*50)
print()
matriz_promedio_seleccionada = matrices_promedio_redimensionadas[digito_seleccionado]
print(f"Matriz Promedio Redimensionada del Dígito {digito_seleccionado}:")
print()
print(matriz_promedio_seleccionada)
print()
print("-"*50)

# --------------------------------- FIN DE LA PARTE A----------------------------------------



# --------------------------------- PARTE B ----------------------------------------

# ---------- EN TODO ESTE BLOQUE, SE BUSCA IMPRIMIR LAS MATRICES MEDIANTE LA BIBLICA "MATPLOTLIB"  ----------

def mostrar_matrices_promedio(matrices_promedio_redimensionadas):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))  # Crear una cuadrícula de subfiguras 2x5
    
    for i in range(2):
        for j in range(5):
            indice = i * 5 + j
            axs[i, j].imshow(matrices_promedio_redimensionadas[indice], cmap='gray')
            axs[i, j].set_title(f'Dígito {indice}')
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

mostrar_matrices_promedio(matrices_promedio_redimensionadas) # Acá se llama a la función.

# --------------------------------- FIN DE LA PARTE B ----------------------------------------



# --------------------------------- PARTE C ---------------------------------------- 

imagen = cv2.imread("datasets/num_1.jpeg", cv2.IMREAD_GRAYSCALE) # Se carga la imagen
imagen_pequeña = cv2.resize(imagen,(8,8)) # La re-escalamos a una matriz de 8x8

# Acá invertimos los valores

for i in range(8):
    for j in range(8):
        imagen_pequeña[i][j] = 255- imagen_pequeña[i][j]
        imagen_pequeña[i][j] = imagen_pequeña[i][j]/255*16

print()
print("Imagen pequeña: ")
print()
print(imagen_pequeña)
print()
print("-"*50)
print()

# --------------------------------- FIN DE LA PARTE C ----------------------------------------



# --------------------------------- PARTE D ---------------------------------------- 

def calcular_distancia_euclidiana(imagen1, imagen2): # Creación de una función para ya no hacer lo mismo Bv, además de que es más entendible
    return np.sqrt(np.sum((imagen1 - imagen2) ** 2))

distancias = []
for i, imagen_dataset in enumerate(imagenes):
    imagen_dataset_redimensionada = cv2.resize(imagen_dataset, (8, 8)).astype(int)
    distancia = calcular_distancia_euclidiana(imagen_pequeña, imagen_dataset_redimensionada)
    distancias.append((distancia, etiquetas[i]))

# Ordenar las distancias y obtener las 3 más pequeñas
distancias.sort()       
dígitos_más_parecidos = distancias[:3]

# --------------------------------- FIN DE LA PARTE D ---------------------------------------- 



# --------------------------------- PARTE E ---------------------------------------- 

print("Los 3 dígitos más parecidos son:")
for index, (distancia, dígito) in enumerate(dígitos_más_parecidos):
    print(f"{index}. Dígito: {dígito}, Distancia: {distancia}")

print()
print("-"*50)
print()


# --------------------------------- FIN DE LA PARTE E ---------------------------------------- 



# --------------------------------- PARTE F ---------------------------------------- 

# Clasificación del nuevo dígito
targets = [dígito for _, dígito in distancias[:10]]

# Contar manualmente las ocurrencias de cada dígito
conteos = {}
for target in targets:
    if target in conteos:
        conteos[target] += 1
    else:
        conteos[target] = 1

# Determinar el dígito más común
dígito_clasificado = None
max_conteo = 0
empate = []

# Acá se determina el dígito más común, además, la lista "empate" solo debería ser == 1 si solo hay un número mayoritario
for dígito, conteo in conteos.items():
    if conteo > max_conteo:
        max_conteo = conteo
        dígito_clasificado = dígito
        empate = [dígito]
    elif conteo == max_conteo:
        empate.append(dígito)
        

dig_mas_parecidos = distancias[:10]

# Esto se inicializa si empate es mayor a 1, eso quiere decir que hubo más de un número con el mismo máximo número de ocurrencias

if len(empate) > 1:
        # Buscar el desempate considerando más distancias
        for distancia, dígito in distancias[10:]:
            dig_mas_parecidos.append((distancia, dígito)) # Se agrega a la lista de dig más parecidos

            if dígito in empate:
                dígito_clasificado = dígito
                break # Se rompe si se encuentra un dígito que está en la lista de los número empatados (porque uno ya sería mayor que el otro)

for index, (distancia, dígito) in enumerate(dig_mas_parecidos):
    print(f"{index}. Dígito: {dígito}, Distancia: {distancia}")

print()
print("-"*50)
print()

print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {dígito_clasificado}")          

# --------------------------------- FIN DE LA PARTE F ---------------------------------------- 



# --------------------------------- PARTE G ----------------------------------------

# Parte g: Comparación con los promedios generados en el inciso a)

distancias_promedio = []
for i, matriz_promedio in enumerate(matrices_promedio_redimensionadas):
    distancia = calcular_distancia_euclidiana(imagen_pequeña, matriz_promedio)
    distancias_promedio.append((distancia, i))

distancias_promedio.sort()

distancia_menor, digito_clasificado_promedio = distancias_promedio[0]

print(f"Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {digito_clasificado_promedio}")
print()

