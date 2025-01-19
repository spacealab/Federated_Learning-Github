import pickle
import numpy as np
import matplotlib.pyplot as plt

# مسیر فایل وزن‌ها را مشخص کنید
weights_path = "global_weights.pkl"

# تابع برای بارگذاری وزن‌ها
def load_weights(weights_path):
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)
    return weights

# تابع برای چاپ کلیدها و اطلاعات اولیه
def print_keys_info(weights):
    print("Available keys in weights:")
    for key in weights.keys():
        print(f"Key: {key}, Shape: {weights[key].shape}")

# تابع برای تجسم وزن‌های یک لایه خاص
def visualize_weights(weights, layer_key, num_filters=5):
    if layer_key not in weights:
        print(f"Key '{layer_key}' not found in weights.")
        return

    layer_weights = weights[layer_key]

    if len(layer_weights.shape) == 4:  # لایه‌های کانولوشنی (4 بعدی)
        print(f"Visualizing {num_filters} filters from layer '{layer_key}'")
        for i in range(min(num_filters, layer_weights.shape[0])):
            filter_weights = layer_weights[i]
            # Combine the filter weights across all input channels
            combined_weights = np.mean(filter_weights, axis=0)
            plt.imshow(combined_weights, cmap="viridis")
            plt.title(f"Filter {i+1} of {layer_key}")
            plt.colorbar()
            plt.show()

    elif len(layer_weights.shape) == 2:  # لایه‌های Dense (2 بعدی)
        print(f"Visualizing weights as heatmap for layer '{layer_key}'")
        plt.imshow(layer_weights, cmap="viridis")
        plt.title(f"Weights of {layer_key}")
        plt.colorbar()
        plt.show()

    else:
        print(f"Unsupported weight shape: {layer_weights.shape}")

# تابع برای تحلیل و چاپ مقادیر آماری وزن‌ها
def analyze_weights_statistics(weights, layer_key):
    if layer_key not in weights:
        print(f"Key '{layer_key}' not found in weights.")
        return

    layer_weights = weights[layer_key]
    mean_value = np.mean(layer_weights)
    std_value = np.std(layer_weights)
    min_value = np.min(layer_weights)
    max_value = np.max(layer_weights)

    print(f"Statistics for layer '{layer_key}':")
    print(f"  Mean: {mean_value}")
    print(f"  Std Dev: {std_value}")
    print(f"  Min: {min_value}")
    print(f"  Max: {max_value}")

# اجرای تحلیل وزن‌ها
if __name__ == "__main__":
    # بارگذاری وزن‌ها
    weights = load_weights(weights_path)

    # نمایش کلیدهای موجود
    print_keys_info(weights)

    # کلید مورد نظر برای تحلیل (تغییر دهید)
    layer_key = "model.22.cv3.2.0.conv.weight"  # به عنوان نمونه

    # تحلیل آماری وزن‌ها
    analyze_weights_statistics(weights, layer_key)

    # تجسم وزن‌ها
    visualize_weights(weights, layer_key, num_filters=5)
