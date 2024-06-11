import torch
import torch.nn as nn
import tqdm


def evaluate(model, loader, criterion, device, testing=False, writer=None):
    # Set the model to evaluation mode
    model.eval()
    loss_list = []
    i = 0
    with torch.no_grad():
        for images, point_clouds in tqdm.tqdm(loader):
            # Move everything to device
            images = images.to(device)
            point_clouds = point_clouds.to(device)

            # Inference
            outputs = model(images)  

            # Compute loss
            loss = criterion(outputs, point_clouds)
            loss_list.append(float(loss))

            # If it is testing phase, write loss for each sample
            if(testing == True):
                writer.add_scalar("Loss per sample - Test", loss, i)

            i += 1

    # Average loss
    eval_loss = sum(loss_list) / len(loss_list)
    return eval_loss
