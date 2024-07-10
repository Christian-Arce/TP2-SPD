import matplotlib.pyplot as plt

# Datos de ejemplo (reemplaza con los resultados reales)
sizes = ['Pequeño', 'Mediano', 'Grande']
sequential_times = [425, 2672, 5182]  # tiempos secuenciales
parallel_times = [71, 426, 862]    # tiempos paralelos
num_threads = 8

# Cálculo de Speedup y Eficiencia
speedup = [seq / par for seq, par in zip(sequential_times, parallel_times)]
efficiency = [s / num_threads for s in speedup]

# Gráfico de Speedup
plt.figure()
plt.plot(sizes, speedup, marker='o', label='Speedup')
plt.xlabel('Tamaño de Datos')
plt.ylabel('Speedup')
plt.title('Speedup del Algoritmo Genético Paralelo')
plt.ylim(0, max(speedup) + 1)  # Extiende el rango del eje y
plt.grid(True)
plt.legend()
plt.savefig('speedup.png')
plt.show()

# Gráfico de Eficiencia
plt.figure()
plt.plot(sizes, efficiency, marker='o', label='Eficiencia')
plt.xlabel('Tamaño de Datos')
plt.ylabel('Eficiencia')
plt.title('Eficiencia del Algoritmo Genético Paralelo')
plt.ylim(0, 1)  # La eficiencia está generalmente en el rango [0, 1]
plt.grid(True)
plt.legend()
plt.savefig('efficiency.png')
plt.show()
