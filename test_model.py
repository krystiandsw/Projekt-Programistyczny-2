import unittest
from data import load_data
from model import build_model
from predict import test_prediction


class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()
        self.model = build_model()
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)

    def test_prediction_output_range(self):
        test_prediction(self.model, self.x_test[0])

    def test_accuracy_above_random(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.assertGreater(acc, 0.1, "Model powinien być lepszy niż losowy (10%).")


if __name__ == '__main__':
    unittest.main()