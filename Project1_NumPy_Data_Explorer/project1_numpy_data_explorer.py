"""
=============================================================
PROJECT 1: NumPy Data Explorer
Syntecxhub Data Science Internship
=============================================================
Topics Covered:
  - Array creation, indexing, slicing
  - Mathematical, axis-wise, and statistical operations
  - Reshaping and broadcasting
  - Save/load operations
  - NumPy vs Python list performance comparison
=============================================================
"""

import numpy as np
import time
import os

print("=" * 60)
print("       PROJECT 1: NumPy Data Explorer")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. ARRAY CREATION
# ─────────────────────────────────────────────
print("\n[1] Array Creation")
print("-" * 40)

arr1d = np.array([10, 20, 30, 40, 50])
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
zeros = np.zeros((3, 3))
ones  = np.ones((2, 4))
rng   = np.arange(0, 20, 2)
linsp = np.linspace(0, 1, 5)
rand  = np.random.randint(1, 100, (4, 4))

print(f"1D Array          : {arr1d}")
print(f"2D Array:\n{arr2d}")
print(f"Zeros (3x3):\n{zeros}")
print(f"Ones  (2x4):\n{ones}")
print(f"arange(0,20,2)    : {rng}")
print(f"linspace(0,1,5)   : {linsp}")
print(f"Random 4x4 int:\n{rand}")

# ─────────────────────────────────────────────
# 2. INDEXING & SLICING
# ─────────────────────────────────────────────
print("\n[2] Indexing & Slicing")
print("-" * 40)

print(f"arr1d[2]          : {arr1d[2]}")
print(f"arr1d[1:4]        : {arr1d[1:4]}")
print(f"arr1d[::-1]       : {arr1d[::-1]}")
print(f"arr2d[1, 2]       : {arr2d[1, 2]}")
print(f"arr2d[:, 1]       : {arr2d[:, 1]}")
print(f"arr2d[0:2, 0:2]:\n{arr2d[0:2, 0:2]}")
print(f"Boolean (arr1d>25): {arr1d[arr1d > 25]}")

# ─────────────────────────────────────────────
# 3. MATHEMATICAL OPERATIONS
# ─────────────────────────────────────────────
print("\n[3] Mathematical Operations")
print("-" * 40)

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a + b             : {a + b}")
print(f"a * b             : {a * b}")
print(f"a ** 2            : {a ** 2}")
print(f"sqrt(b)           : {np.sqrt(b)}")
print(f"log(b)            : {np.round(np.log(b), 3)}")
print(f"sin(a)            : {np.round(np.sin(a), 3)}")
print(f"dot product a·b   : {np.dot(a, b)}")

matrix = np.array([[2, 1], [5, 3]])
print(f"Matrix inverse:\n{np.linalg.inv(matrix)}")

# ─────────────────────────────────────────────
# 4. AXIS-WISE & STATISTICAL OPERATIONS
# ─────────────────────────────────────────────
print("\n[4] Axis-wise & Statistical Operations")
print("-" * 40)

data = np.array([[4, 7, 2], [1, 9, 5], [8, 3, 6]])
print(f"Dataset:\n{data}")
print(f"Sum (all)         : {np.sum(data)}")
print(f"Sum (axis=0, col) : {np.sum(data, axis=0)}")
print(f"Sum (axis=1, row) : {np.sum(data, axis=1)}")
print(f"Mean (all)        : {np.mean(data):.2f}")
print(f"Mean (axis=0)     : {np.mean(data, axis=0)}")
print(f"Median            : {np.median(data)}")
print(f"Std Dev           : {np.std(data):.2f}")
print(f"Variance          : {np.var(data):.2f}")
print(f"Min / Max         : {np.min(data)} / {np.max(data)}")
print(f"ArgMin / ArgMax   : {np.argmin(data)} / {np.argmax(data)}")
print(f"Percentile 25/75  : {np.percentile(data, 25)} / {np.percentile(data, 75)}")

# ─────────────────────────────────────────────
# 5. RESHAPING & BROADCASTING
# ─────────────────────────────────────────────
print("\n[5] Reshaping & Broadcasting")
print("-" * 40)

arr = np.arange(12)
print(f"Original (12,)    : {arr}")
r1 = arr.reshape(3, 4)
print(f"Reshaped (3,4):\n{r1}")
r2 = arr.reshape(2, 2, 3)
print(f"Reshaped (2,2,3):\n{r2}")
print(f"Flattened         : {r1.flatten()}")
print(f"Transposed (3,4)->(4,3):\n{r1.T}")

# Broadcasting
col_vec = np.array([[1], [2], [3]])   # shape (3,1)
row_vec = np.array([10, 20, 30, 40]) # shape (4,)
broadcast_result = col_vec + row_vec  # broadcasts to (3,4)
print(f"\nBroadcasting (3,1) + (4,) -> (3,4):\n{broadcast_result}")

# ─────────────────────────────────────────────
# 6. SAVE / LOAD OPERATIONS
# ─────────────────────────────────────────────
print("\n[6] Save & Load Operations")
print("-" * 40)

os.makedirs("numpy_output", exist_ok=True)

# Single array — .npy
np.save("numpy_output/array_2d.npy", arr2d)
loaded_single = np.load("numpy_output/array_2d.npy")
print(f"Saved & loaded array_2d.npy:\n{loaded_single}")

# Multiple arrays — .npz
np.savez("numpy_output/multi_arrays.npz", arr_a=a, arr_b=b, data=data)
loaded_multi = np.load("numpy_output/multi_arrays.npz")
print(f"Loaded from .npz -> keys: {list(loaded_multi.keys())}")
print(f"  arr_a: {loaded_multi['arr_a']}")

# Text CSV
np.savetxt("numpy_output/data_matrix.csv", data, delimiter=",", fmt="%d",
           header="col1,col2,col3", comments="")
loaded_txt = np.loadtxt("numpy_output/data_matrix.csv", delimiter=",", skiprows=1)
print(f"Saved & loaded CSV:\n{loaded_txt}")

print("\nFiles saved in ./numpy_output/")

# ─────────────────────────────────────────────
# 7. NUMPY vs PYTHON LIST PERFORMANCE
# ─────────────────────────────────────────────
print("\n[7] NumPy vs Python List Performance")
print("-" * 40)

SIZE = 1_000_000

py_list = list(range(SIZE))
np_arr  = np.arange(SIZE)

# Sum
t0 = time.perf_counter(); s1 = sum(py_list);        t_list_sum = time.perf_counter() - t0
t0 = time.perf_counter(); s2 = np.sum(np_arr);      t_np_sum   = time.perf_counter() - t0

# Element-wise multiply
t0 = time.perf_counter()
result_list = [x * 2 for x in py_list]
t_list_mul = time.perf_counter() - t0

t0 = time.perf_counter()
result_np = np_arr * 2
t_np_mul = time.perf_counter() - t0

print(f"{'Operation':<30} {'Python List':>14} {'NumPy':>14} {'Speedup':>10}")
print("-" * 70)
print(f"{'Sum (1M elements)':<30} {t_list_sum*1000:>12.2f}ms {t_np_sum*1000:>12.2f}ms {t_list_sum/t_np_sum:>9.1f}x")
print(f"{'Multiply x2 (1M elements)':<30} {t_list_mul*1000:>12.2f}ms {t_np_mul*1000:>12.2f}ms {t_list_mul/t_np_mul:>9.1f}x")

print("\n✅ Memory comparison (1M integers):")
import sys
print(f"   Python list size : {sys.getsizeof(py_list):,} bytes")
print(f"   NumPy array size : {np_arr.nbytes:,} bytes")
print(f"   Memory savings   : {(1 - np_arr.nbytes/sys.getsizeof(py_list))*100:.1f}%")

print("\n" + "=" * 60)
print("  Project 1 Complete — NumPy Data Explorer ✅")
print("=" * 60)
