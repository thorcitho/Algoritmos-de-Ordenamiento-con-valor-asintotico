import random
import timeit
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import time  # Agregar esta línea para importar el módulo time
import numpy as np
# Implementación de los algoritmos de ordenamiento

# Iterativos
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[i] < arr[left]:
            largest = left

        if right < n and arr[largest] < arr[right]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def counting_sort(arr):
    max_value = max(arr)
    min_value = min(arr)
    range_of_elements = max_value - min_value + 1
    count = [0] * range_of_elements
    output = [0] * len(arr)

    for i in range(len(arr)):
        count[arr[i] - min_value] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_value] - 1] = arr[i]
        count[arr[i] - min_value] -= 1

    for i in range(len(arr)):
        arr[i] = output[i]

def radix_sort(arr):
    max_value = max(arr)
    exp = 1
    while max_value // exp > 0:
        counting_sort_radix(arr, exp)
        exp *= 10

def counting_sort_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def bucket_sort(arr):
    max_value = max(arr)
    min_value = min(arr)
    bucket_size = (max_value - min_value) / len(arr)
    buckets = [[] for _ in range(len(arr))]

    for num in arr:
        index = int((num - min_value) / bucket_size)
        buckets[index].append(num)

    for i in range(len(arr)):
        insertion_sort(buckets[i])

    k = 0
    for i in range(len(arr)):
        for j in range(len(buckets[i])):
            arr[k] = buckets[i][j]
            k += 1

# Recursivos
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i, j, k = 0, 0, 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

# Lista de métodos de ordenamiento
methods = [
    ("Selection Sort", selection_sort),
    ("Bubble Sort", bubble_sort),
    ("Insertion Sort", insertion_sort),
    ("Heap Sort", heap_sort),
    ("Counting Sort", counting_sort),
    ("Radix Sort", radix_sort),
    ("Bucket Sort", bucket_sort),
    ("Merge Sort", merge_sort),
    ("Quick Sort", quick_sort)
]

# Función para medir el tiempo de ejecución
def measure_time(method, arr):
    time = timeit.timeit(f"{method}({arr})", globals=globals(), number=10)  # Ajustamos el número de iteraciones a 10
    return time

# Función para generar los datos
def generate_data(method_name, element_count):
    methods = [
        ("Selection Sort", selection_sort),
        ("Bubble Sort", bubble_sort),
        ("Insertion Sort", insertion_sort),
        ("Heap Sort", heap_sort),
        ("Counting Sort", counting_sort),
        ("Radix Sort", radix_sort),
        ("Bucket Sort", bucket_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort)
    ]

    method = None
    for name, func in methods:
        if name == method_name:
            method = func
            break

    if method is None:
        print(f"El método '{method_name}' no existe.")
        return

    element_counts = []
    times = []

    for i in range(element_count):
        arr = [random.randint(0, 10000) for _ in range(i * 50 + 100)]
        time = measure_time(method.__name__, arr)  # Pasamos 'arr' como argumento a measure_time
        element_counts.append(i * 50 + 100)
        times.append(time)

    return element_counts, times

# Función para obtener la notación asintótica
def get_notation_asymptotic(x, y):
    coefficients = np.polyfit(np.log(x), y, 1)
    a, b = coefficients[0], coefficients[1]

    if abs(a) > 1e-6:
        notation = f"{a:.5f} * log(n) + {b:.5f}"
    else:
        notation = f"{b:.5f}"

    return notation

# Función para generar los datos y graficarlos
def generate_and_plot_data(method_name, element_count):
    methods = [
        ("Selection Sort", selection_sort),
        ("Bubble Sort", bubble_sort),
        ("Insertion Sort", insertion_sort),
        ("Heap Sort", heap_sort),
        ("Counting Sort", counting_sort),
        ("Radix Sort", radix_sort),
        ("Bucket Sort", bucket_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort)
    ]

    method = None
    for name, func in methods:
        if name == method_name:
            method = func
            break

    if method is None:
        print(f"El método '{method_name}' no existe.")
        return

    element_counts, times = generate_data(method_name, element_count)
    notation = get_notation_asymptotic(element_counts, times)

    # Limpiar la figura antes de trazar los nuevos datos
    plt.clf()

    # Subplot para el gráfico de tiempo de ejecución
    plt.subplot(2, 1, 1)
    plt.plot(element_counts, times, marker='o', label=method_name)
    plt.xlabel("Cantidad de Elementos")
    plt.ylabel("Tiempo (segundos)")
    plt.title(f"Tiempo de Ejecución de {method_name}")
    plt.grid(True)
    plt.legend(loc='upper left')

    # Subplot para el gráfico de notación asintótica
    plt.subplot(2, 1, 2)
    plt.plot(element_counts, times, marker='o', label="Tiempo de Ejecución")
    plt.plot(element_counts, np.polyval(np.polyfit(np.log(element_counts), times, 1), np.log(element_counts)), label="Notación Asintótica")
    plt.xlabel("Cantidad de Elementos")
    plt.ylabel("Tiempo (segundos)")
    plt.title(f"Notación Asintótica de {method_name}")
    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()  # Ajustar los subplots para evitar superposiciones
    plt.pause(0.001)  # Pausa para actualizar la gráfica   

# Función para manejar el evento del botón
def on_button_click():
    selected_method = method_combo.get()
    for i in range(1, 201):  # Incrementar gradualmente hasta 10000 elementos
        generate_and_plot_data(selected_method, i)
        time.sleep(1)

# Creación de la interfaz gráfica
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Algoritmos de Ordenamiento")
    root.geometry("400x200")

    method_combo = ttk.Combobox(root, values=[method[0] for method in methods])
    method_combo.set("Selecciona un algoritmo")
    method_combo.pack(pady=20)

    btn_generate = ttk.Button(root, text="Generar Gráfico", command=on_button_click)
    btn_generate.pack()

    plt.ion()  # Activar modo interactivo
    plt.show()

    root.mainloop()
