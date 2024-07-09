# No recuerdo lo que hicimos en clase así que estoy viendo las grabaciones
# Si alguien quiere ver mis comentarios, entonces puede leer todo lo que hago aquí.

import cv2
from sklearn import datasets

# Esto no es un diccionario como tal, la clave es un atributos, y los valores serían igualmente valores.

# La primera clave asociada al diccionario es una lista de listas

digits = datasets.load_digits()

# Cada lista dentro de la lista de listas es una matriz aplanada
print("\nLa primera matriz aplanada:")
print(digits["data"][0])

print("\nLa última matriz aplanada:")
print(digits["data"][-1])

# La siguiente clave es "target" (se pronuncia a algo parecido a taergoet), 