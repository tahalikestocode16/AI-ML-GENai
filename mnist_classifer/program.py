import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# =========================
# 1. Load MNIST Dataset
# =========================

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# =========================
# 2. Create DataLoaders
# =========================

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# =========================
# 3. Create Neural Network
# =========================

class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


model = DigitClassifier()


# =========================
# 4. Loss Function
# =========================

loss_function = nn.CrossEntropyLoss()


# =========================
# 5. Optimizer
# =========================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 6. Training
# =========================

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = loss_function(predictions, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f}"
    )


# =========================
# 7. Test Model
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        predicted_classes = torch.argmax(
            predictions,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted_classes == labels
        ).sum().item()


accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# =========================
# 8. Test One Image
# =========================

image, actual_label = test_data[0]

with torch.no_grad():

    output = model(image.unsqueeze(0))

    predicted_label = torch.argmax(
        output,
        dim=1
    ).item()
    
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f'Actual label: {actual_label} | Predicted Label: {predicted_label}')
plt.axis("off")
plt.show()

print("Actual:", actual_label)
print("Predicted:", predicted_label)