from __future__ import absolute_import, division, print_function, unicode_literals #Para compatibilidad entre python 2 y 3

import tensorflow as tf #Importar tensorflow 
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt #Para graficar
import logging #Para mensajes de error
logger = tf.get_logger() #Para mensajes de error

logger.setLevel(logging.ERROR) #Para mensajes de error


dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True) #Cargar el dataset de numeros manuscritos de MNIST en dos partes, datos y metadatos
train_dataset, test_dataset = dataset['train'], dataset['test'] #Separar los datos de entrenamiento y pruebas en dos variables

class_names = [ #Clases de numeros
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis', 
    'Siete', 'Ocho', 'Nueve'
]

num_train_examples = metadata.splits['train'].num_examples #Numero de ejemplos de entrenamiento en el dataset es decir 60000
num_test_examples = metadata.splits['test'].num_examples #Numero de ejemplos de pruebas en el dataset es decir 10000

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def normalize(images, labels): #Funcion para normalizar las imagenes
    images = tf.cast(images, tf.float32) #Convertir las imagenes a flotantes
    images /= 255 #Dividir las imagenes entre 255
    return images, labels #Retornar las imagenes y las etiquetas

train_dataset = train_dataset.map(normalize) #Normalizar las imagenes de entrenamiento es decir convertir las imagenes a flotantes y dividirlas entre 255
test_dataset = test_dataset.map(normalize) #Normalizar las imagenes de pruebas es decir convertir las imagenes a flotantes y dividirlas entre 255

#Estructura de la red
model = tf.keras.Sequential([ #Modelo secuencial
	tf.keras.layers.Flatten(input_shape=(28,28,1)), #Capa de entrada 
	tf.keras.layers.Dense(64, activation=tf.nn.relu), #Capa oculta de 64 neuronas es decir 64 nodos
	tf.keras.layers.Dense(64, activation=tf.nn.relu), #Capa oculta de 64 neuronas es decir 64 nodos
	tf.keras.layers.Dense(10, activation=tf.nn.softmax) #para clasificacion de 10 clases es decir 10 nodos con funcion de activacion softmax
])

#Indicar las funciones a utilizar
model.compile( #Compilar el modelo
	optimizer='adam', #Optimizador adam
	loss='sparse_categorical_crossentropy', #Funcion de perdida entropia cruzada
	metrics=['accuracy'] #Metrica de precision
)

#Aprendizaje por lotes de 32 cada lote es decir 32 imagenes y etiquetas a la vez
BATCHSIZE = 32 #Tamaño del lote
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE) #Repetir, mezclar y dividir en lotes
test_dataset = test_dataset.batch(BATCHSIZE) #Dividir en lotes

#Realizar el aprendizaje
model.fit( #Entrenar el modelo
	train_dataset, epochs=5, #Entrenar el modelo con 5 epocas
	steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE) #No sera necesario pronto
)

#Evaluar nuestro modelo ya entrenado, contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate( #esto nos da la precision del modelo
	test_dataset, steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas: ", test_accuracy) #Imprimir la precision del modelo


for test_images, test_labels in test_dataset.take(1): #Tomar un lote de imagenes y etiquetas de pruebas
	test_images = test_images.numpy() #Convertir las imagenes a arreglos de numpy
	test_labels = test_labels.numpy() #Convertir las etiquetas a arreglos de numpy
	predictions = model.predict(test_images) #Predecir las imagenes de pruebas

def plot_image(i, predictions_array, true_labels, images): #Funcion para graficar las imagenes
	predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img[...,0], cmap=plt.cm.binary) #Mostrar la imagen en escala de grises

	predicted_label = np.argmax(predictions_array) #Obtener la prediccion
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color) #Mostrar la prediccion

def plot_value_array(i, predictions_array, true_label): #Funcion para graficar los valores de las predicciones
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#888888")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

numrows=5 
numcols=3
numimages = numrows*numcols

plt.figure(figsize=(2*2*numcols, 2*numrows)) #Tamaño de la figura
for i in range(numimages): #Graficar las imagenes
	plt.subplot(numrows, 2*numcols, 2*i+1) #Graficar la imagen
	plot_image(i, predictions, test_labels, test_images) #Graficar la imagen
	plt.subplot(numrows, 2*numcols, 2*i+2) #Graficar los valores de las predicciones
	plot_value_array(i, predictions, test_labels) #Graficar los valores de las predicciones

plt.show() #Mostrar las graficas


"""
Datos son la información primaria que se utiliza directamente, como las imágenes de dígitos y 
sus etiquetas en el dataset MNIST. Metadatos son datos sobre esos datos, proporcionando información 
adicional como la cantidad total de ejemplos y las dimensiones de las imágenes. Por ejemplo, en el dataset MNIST, 
los datos incluyen las imágenes y etiquetas de los dígitos manuscritos, mientras que los metadatos informan sobre el 
número de ejemplos en los conjuntos de entrenamiento y prueba, así como las características del dataset
"""