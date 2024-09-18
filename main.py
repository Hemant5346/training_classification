
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from collections import Counter


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    return corrects.double() / len(labels)


def main():
    # Define transformations for the training set (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define transformations for the validation set
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    data_dir = '/Users/hemantgoyal/Desktop/final_training_hair/training_classification/output_folder'  # Path to your dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # Split into train and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Update transforms for validation dataset
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Calculate class weights for imbalanced dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts = Counter(dataset.targets)
    total_samples = float(sum(class_counts.values()))
    class_weights = [total_samples / class_counts[i] for i in range(len(dataset.classes))]
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(dataset.classes))  # Adjust for your number of classes
    model = model.to(device)

    # Define the loss function (weighted for imbalanced dataset) and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop with validation
    num_epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0  # Track the best validation accuracy for saving the model

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_train += calculate_accuracy(outputs, labels).item() * inputs.size(0)

        epoch_train_loss = running_loss / train_size
        epoch_train_acc = correct_train / train_size
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                correct_val += calculate_accuracy(outputs, labels).item() * inputs.size(0)

        epoch_val_loss = running_loss / val_size
        epoch_val_acc = correct_val / val_size
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Update the learning rate
        scheduler.step()

        # Save the model with the best validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_hair_color_classification_model.pth')
   


if __name__ == '__main__':
    main()
