import numpy as np
from sklearn.datasets import load_digits
import cv2
import matplotlib.pyplot as plt

digits = load_digits()

imagenes = digits.images
etiquetas = digits.target

matrices_promedio_redimensionadas = []
for digito in range(10):
    imagenes_digito = imagenes[etiquetas == digito]
    
    matriz_promedio = np.mean(imagenes_digito, axis=0)
    
    matriz_promedio_redimensionada = cv2.resize(matriz_promedio, (8, 8), interpolation=cv2.INTER_AREA).astype(int)
    
    matrices_promedio_redimensionadas.append(matriz_promedio_redimensionada)

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

mostrar_matrices_promedio(matrices_promedio_redimensionadas)

digito_seleccionado = int(input("Ingrese el dígito del cual desea ver la matriz promedio (0-9): "))

matriz_promedio_seleccionada = matrices_promedio_redimensionadas[digito_seleccionado]

print(f"Matriz Promedio Redimensionada del Dígito {digito_seleccionado}:")
print(matriz_promedio_seleccionada)


# Esto permite poner una imagen en el paréntesis, el cv2.IMREAD_GRAYSCALE lo lleva a escala de grises

imagen = cv2.imread("datasets/imagen.png", cv2.IMREAD_GRAYSCALE)

# .resize escala la matriz de imagen en 8 x 8

imagen_pequeña = cv2.resize(imagen,(8,8))

# Convertimos cada número de la matriz de 0 a 255 e inverso.

i = 0
while i<=7:
    j = 0
    while j <= 7:
        imagen_pequeña[i][j] = 255- imagen_pequeña[i][j]
        j += 1
    i += 1

# En estos bucles hacemos el martillo para que se rescale de 255 a 16

i = 0
while i<=7:
    j = 0
    while j <= 7:
        imagen_pequeña[i][j] = imagen_pequeña[i][j]/255*16
        j += 1
    i += 1

print()
print(imagen_pequeña)

# Calcular la distancia euclidiana y encontrar los 3 dígitos más cercanos

def calcular_distancia_euclidiana(imagen1, imagen2):
    return np.sqrt(np.sum((imagen1 - imagen2) ** 2))

distancias = []
for i, imagen_dataset in enumerate(imagenes):
    imagen_dataset_redimensionada = cv2.resize(imagen_dataset, (8, 8)).astype(int)
    distancia = calcular_distancia_euclidiana(imagen_pequeña, imagen_dataset_redimensionada)
    distancias.append((distancia, etiquetas[i]))

# Ordenar las distancias y obtener las 3 más pequeñas
distancias.sort()
dígitos_más_parecidos = distancias[:3]

print("Los 3 dígitos más parecidos son:")
for distancia, dígito in dígitos_más_parecidos:
    print(f"Dígito: {dígito}, Distancia: {distancia}")

# Clasificación del nuevo dígito
targets = [dígito for _, dígito in dígitos_más_parecidos]

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
for dígito, conteo in conteos.items():
    if conteo > max_conteo:
        max_conteo = conteo
        dígito_clasificado = dígito

# Verificar si hay 2 o 3 dígitos iguales
if max_conteo >= 2:
    print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {dígito_clasificado}")
else:
    # Si los 3 targets son diferentes, tomamos el dígito con la menor distancia como clasificación
    dígito_clasificado = dígitos_más_parecidos[0][1]
    print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {dígito_clasificado}")