import matplotlib.pyplot as plt

# Datos proporcionados
sequential_times = [277, 1572, 3217]  # tiempos secuenciales

# Tiempos paralelos para diferentes números de procesos
parallel_times_4 = [71, 420, 820]  # tiempos paralelos para 4 procesos
parallel_times_8 = [36, 204, 420]  # tiempos paralelos para 8 procesos
parallel_times_16 = [33, 200, 410]  # tiempos paralelos para 16 procesos

# Números de procesos
num_processes = [4, 8, 16]

# Cálculo de Speedup y Eficiencia
speedup_4 = [sequential_times[i] / parallel_times_4[i] for i in range(len(sequential_times))]
efficiency_4 = [speedup / 4 for speedup in speedup_4]

speedup_8 = [sequential_times[i] / parallel_times_8[i] for i in range(len(sequential_times))]
efficiency_8 = [speedup / 8 for speedup in speedup_8]

speedup_16 = [sequential_times[i] / parallel_times_16[i] for i in range(len(sequential_times))]
efficiency_16 = [speedup / 16 for speedup in speedup_16]

# Gráfico de Speedup
plt.figure(figsize=(12, 6))

# Speedup para diferentes tamaños de problema
plt.plot(num_processes, [speedup_4[0], speedup_8[0], speedup_16[0]], marker='x', label='Pequeño', color='b')
plt.plot(num_processes, [speedup_4[1], speedup_8[1], speedup_16[1]], marker='s', label='Mediano', color='r')
plt.plot(num_processes, [speedup_4[2], speedup_8[2], speedup_16[2]], marker='o', label='Grande', color='g')

plt.xlabel('Número de Procesos', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup del Algoritmo Genético Paralelo - MPI', fontsize=14)
plt.xticks(num_processes)
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de Eficiencia
plt.figure(figsize=(12, 6))

# Eficiencia para diferentes tamaños de problema
plt.plot(num_processes, [efficiency_4[0], efficiency_8[0], efficiency_16[0]], marker='x', label='Pequeño', color='b')
plt.plot(num_processes, [efficiency_4[1], efficiency_8[1], efficiency_16[1]], marker='s', label='Mediano', color='r')
plt.plot(num_processes, [efficiency_4[2], efficiency_8[2], efficiency_16[2]], marker='o', label='Grande', color='g')

plt.xlabel('Número de Procesos', fontsize=12)
plt.ylabel('Eficiencia', fontsize=12)
plt.title('Eficiencia del Algoritmo Genético Paralelo - MPI', fontsize=14)
plt.xticks(num_processes)
plt.ylim(0, 1.1)  # Eficiencia usualmente entre 0 y 1
plt.legend()
plt.grid(True)
plt.show()
