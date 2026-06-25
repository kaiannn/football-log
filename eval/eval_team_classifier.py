def evaluate_team_classifier(model, data_loader, device, metrics, warmup_frames=50, stable_frames=50):
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
            if warmup_frames > 0 and total > warmup_frames:
                break
        # ... 中间逻辑 ...

    accuracy = correct / total
    return accuracy
