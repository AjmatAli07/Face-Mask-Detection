import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from utils import transform


# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# Load dataset
dataset = ImageFolder(
    root="dataset",
    transform=transform
)


# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(
    dataset,
    [train_size, test_size]
)


# Data loaders
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)


# CNN model
class MaskDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                3,16,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,32,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,64,3,padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64*16*16,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self,x):

        x=self.conv(x)
        x=self.fc(x)

        return x



model=MaskDetector().to(device)

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs=20

best_accuracy=0


for epoch in range(epochs):

    model.train()

    running_loss=0

    for images,labels in train_loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()



    # Validation
    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for images,labels in test_loader:

            images=images.to(device)

            labels=labels.to(device)

            outputs=model(images)

            _,predicted=torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted==labels
            ).sum().item()



    accuracy=100*correct/total


    print(
        f"Epoch {epoch+1}/{epochs}"
    )

    print(
        f"Loss: {running_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    print("-"*40)



    # Save only best model
    if accuracy > best_accuracy:

        best_accuracy=accuracy

        torch.save(
            model.state_dict(),
            "models/mask_detector.pth"
        )

        print("Best model saved")


print(
    f"Training Complete"
)

print(
    f"Best Accuracy: {best_accuracy:.2f}%"
)