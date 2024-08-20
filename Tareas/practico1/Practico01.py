import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Semilla para asegurar reproducibilidad
np.random.seed(42) # para que mis datos sean los misma en cada ejecución

# Generar 100 estaturas entre 1.50 y 2.00 metros
estaturas = np.random.uniform(1.50, 2.00, 100)
estaturas = np.round(estaturas, 2)

# Generar pesos basados en la estatura, asegurando coherencia
pesos = []
for estatura in estaturas:
    # Peso base es proporcional a la estatura más un poco de variabilidad usando la fomula de la IMC (Índice de Masa Corporal) = peso / estatura^2
    peso = np.random.uniform(18.5, 25.0) * (estatura ** 2) # enfoque cuadrático es decir, el peso crece más rápidamente a medida que aumenta la estatura 18.5 y 25.0 son los valores de peso mínimo y máximo respectivamente
    pesos.append(round(peso, 2)) # redondeamos a 2 decimales y lo agregamos a la lista de pesos

# Crear DataFrame con los datos generados
df = pd.DataFrame({
    'Estatura (m)': estaturas,
    'Peso (kg)': pesos
})

# Guardar en un archivo CSV para revisarlo si es necesario
df.to_csv('Tareas/practico1/datos_generados.csv', index=False)

# Imprimir los primeros 5 ejemplos para revisar
print(df.head())



# Calcular la pendiente (m) y la intersección (b) de la recta y = mx + b
x = df['Estatura (m)']
y = df['Peso (kg)']
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2) 
b = np.mean(y) - m * np.mean(x)

# Crear los valores de y basados en la fórmula de la recta
y_line = m * x + b

plt.scatter(df['Estatura (m)'], df['Peso (kg)'], color='blue', label='Datos')
plt.plot(x, y_line, color='red', label='Línea ajustada')
plt.title('Estatura vs Peso con Línea Ajustada')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()




"""
Enfoque Curva Cuadrática:
Ejemplo: Imagina que estás calculando cuánto pesa un pastel en función de su tamaño. Si un pastel pequeño pesa poco y un pastel grande
 pesa mucho más (no solo un poco más, sino significativamente más), entonces hay una relación cuadrática entre el tamaño y el peso. 
 En este caso, el peso crece más rápidamente a medida que aumenta el tamaño.
Enfoque lineal (Rango Saludable):
Ejemplo: Ahora imagina que estás eligiendo un peso ideal para diferentes alturas de personas, usando una tabla de peso saludable. 
Para cada altura, seleccionas un peso al azar dentro de un rango saludable. Aquí, sigues una relación controlada que asegura que los 
pesos son realistas y saludables para cada altura.
"""