# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the MNIST dataset. The MNIST dataset contains handwritten digit images (0-9), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

![Screenshot 2025-03-21 140451](https://github.com/user-attachments/assets/66050e28-6366-441c-b8c7-becfe9e26291)


## DESIGN STEPS

### STEP 1:
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

### STEP 2:
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

### STEP 3:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 4:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

### STEP 5:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 6:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7:
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM
### Name: DIVYA DHARSHINI R
### Register Number: 212223040042
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: DIVYA DHARSHINI R')
        print('Register Number: 212223040042      ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/2ac6d575-969a-4552-9956-ef7e31835728)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/7c75d48d-5ce3-460c-84d0-b84c04de8eb9)


### Classification Report

![image](https://github.com/user-attachments/assets/abf7b017-3250-4ca9-9f47-815cbc0e41f0)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d875c845-cdfb-4b62-a1e6-bcb200bb22c0)
 

## RESULT
Thus the development of a convolutional deep neural network for image classification is executed successfully.
