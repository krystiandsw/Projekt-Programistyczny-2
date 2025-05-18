from data import load_data
from model import build_model
from predict import show_predictions, test_prediction


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    print("\n--- Trenowanie modelu ---")
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    print("\n--- Ewaluacja ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Dokładność na zbiorze testowym: {test_acc:.4f}")

    print("\n--- Przykładowe predykcje ---")
    show_predictions(model, x_test, y_test)

    print("\n--- Test jednostkowy ---")
    test_prediction(model, x_test[0])
    print("Test jednostkowy OK.")
