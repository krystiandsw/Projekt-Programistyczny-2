import torch
import matplotlib.pyplot as plt
import unittest
from model import SimpleNN

def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

def show_predictions(model, dataloader, num_images=6):
    model.eval()
    images, labels = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 2))
    for i in range(num_images):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {preds[i].item()}")
        axes[i].axis("off")
    plt.show()

class TestSimpleNN(unittest.TestCase):
    def test_forward_pass(self):
        model = SimpleNN()
        # Create a dummy input tensor with batch size 4 and image size 28x28
        input_tensor = torch.randn(4, 1, 28, 28)
        output = model(input_tensor)
        # Output should have shape (4, 10) for 10 classes
        self.assertEqual(output.shape, (4, 10))

if __name__ == "__main__":
    unittest.main()

