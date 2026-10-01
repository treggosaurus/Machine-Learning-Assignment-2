from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def plot_model(model, x_testing_set, y_testing_set):
    predictions = model.predict(x_testing_set)

    error = mean_absolute_error(y_testing_set, predictions)
    print(f"Mean Absolute Error: {error} MW")

    plt.figure(figsize=(12, 5))
    plt.plot(y_testing_set.values[:100], label="Actual Demand")
    plt.plot(predictions[:100], label="Predicted Demand")
    plt.ylabel("Megawatts (MW)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()

def plot_predictions(y_testing_set, base_predictions, safe_predictions):
    base_error = mean_absolute_error(y_testing_set, base_predictions)
    safe_error = mean_absolute_error(y_testing_set, safe_predictions)

    print(f"Base Model Mean Absolute Error: {base_error:.2f} MW")
    print(f"Safe Model (Buffered) Mean Absolute Error: {safe_error:.2f} MW")

    plt.figure(figsize=(12, 5))
    plt.plot(y_testing_set.values[:100], label="Actual Demand", color='#1F77B4', linewidth=2)
    plt.plot(base_predictions[:100], label="Base Prediction", color='#FF7F0E', linestyle='--')
    plt.plot(safe_predictions[:100], label="Safe Prediction (+1.5 Std Dev)", color='#15BA07', linewidth=2)

    plt.title("Actual Demand vs. Base and Safe Predictions")
    plt.ylabel("Megawatts (MW)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
