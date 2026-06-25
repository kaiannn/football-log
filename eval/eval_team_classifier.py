def evaluate_team_classifier(model, data_loader, device, epoch, warmup=False, stable=False):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (predicted == labels).sum().item()
            if warmup and epoch < 50:
                continue
            if stable and epoch >= 50:
                continue
    accuracy = correct / total
    return accuracy
