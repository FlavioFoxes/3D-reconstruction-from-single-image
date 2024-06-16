import torch
import torch.nn as nn
import tqdm

"""
Evaluation function:
it tests the model on the loader. It is used for both evaluation stage (during training) and test stage.

Arguments:
        model:                      model to evaluate
        loader (DataLoader):        DataLoader on which the model is evaluated
        criterion (Loss):           loss function
        device:                     device on which everything sit
        testing (bool):             True if we are in test stage, False otherwise
        writer (SummaryWriter):     if testing is True and writer is passed as argument,
                                    it writes logs
"""
def evaluate(model, loader, criterion, device, testing=False, writer=None):
    # Set the model to evaluation mode
    model.eval()
    loss_list = []
    i = 0
    with torch.no_grad():
        # For each sample in the evaluation/test loader
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
