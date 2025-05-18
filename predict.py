import numpy as np
import matplotlib.pyplot as plt


def show_predictions(model, x_test, y_test, n=5):
    predictions = model.predict(x_test[:n])
    for i in range(n):
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Przewidywana: {np.argmax(predictions[i])} | Faktyczna: {np.argmax(y_test[i])}")
        plt.axis('off')
        plt.show()


def test_prediction(model, x_sample):
    sample = np.expand_dims(x_sample, axis=0)
    prediction = np.argmax(model.predict(sample))
    assert 0 <= prediction <= 9, "Nieprawidłowa predykcja"