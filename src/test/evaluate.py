import torch
import torch.nn as nn
import tqdm


def evaluate(model, loader, criterion, device, testing=False, writer=None):
    model.eval()  # Set the model to evaluation mode
    loss_list = []
    i = 0
    with torch.no_grad():
        for images, point_clouds in tqdm.tqdm(loader):
            images = images.to(device)
            point_clouds = point_clouds.to(device)

            outputs = model(images)  # Forward pass

            loss = criterion(outputs, point_clouds)  # Compute loss
            loss_list.append(float(loss))

            if(testing == True):
                writer.add_scalar("Loss per sample - Test", loss, i)

            i += 1

    eval_loss = sum(loss_list) / len(loss_list)  # Compute evaluation loss
    return eval_loss
