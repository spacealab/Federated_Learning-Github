import pickle

file_path = "./global_weights.pkl"

with open(file_path, 'rb') as f:
    global_weights = pickle.load(f)

for key, value in global_weights.items():
    print(f"Key: {key}")
    print(f"Shape: {value.shape if hasattr(value, 'shape') else 'Scalar'}")
    print(f"Values: {value[:5] if hasattr(value, '__len__') else value}")  # نمایش اولین مقادیر (در صورت وجود)
    print("-" * 50)
