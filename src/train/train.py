import sys
from src.utils.utils import save_checkpoint
import matplotlib.pyplot as plt
import tqdm
import matplotlib.pyplot as plt
import torch

from src.test.evaluate import *


"""
Train function for one epoch:
it trains the model on the loader for one epoch

Arguments:
        model:                      model to evaluate
        train_loader (DataLoader):  DataLoader on which the model is trained
        device:                     device on which everything sit
        optimizer:                  optimization algorithm
        criterion (Loss):           loss function
        epoch:                      current epoch
        writer (SummaryWriter):     it writes logs
"""
def trainOneEpoch(model, train_loader, device, optimizer, criterion, epoch, writer):
    # Set the model to training mode
    model.train()  

    loss_list = []
    i = 0

    # For each sample in the train loader
    for images, point_clouds in tqdm.tqdm(train_loader):
        # Move everything to device
        images = images.to(device)
        point_clouds = point_clouds.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()  

        # Forward pass
        outputs = model(images) 
        
        # Compute loss
        loss = criterion(outputs, point_clouds)

        # Maps the normalized indexd on the x axis of TensorBoard between 0 and total_epochs
        writer_idx = (epoch + i / 4)
        writer.add_scalar("Loss per step - Train", loss, writer_idx)

        # Bacward pass
        loss.backward() 

        # Optimization step
        optimizer.step()  

        # Append current loss to the list
        loss_list.append(float(loss))
        i += 1
        
        # Print current loss
        print(f'---Running Loss: {float(loss):.4f}')

    # Compute epoch loss
    epoch_loss = sum(loss_list) / len(loss_list) 
    return epoch_loss

"""
Train function:
it manages training phase for every epoch

Arguments:
        model:                      model to evaluate
        train_loader (DataLoader):  DataLoader on which the model is trained
        eval_loader (DataLoader):   DataLoader on which the model is evaluated
        optimizer:                  optimization algorithm
        criterion (Loss):           loss function
        device:                     device on which everything sit
        num_epoch:                  total number of epochs
        writer (SummaryWriter):     it writes logs
        patience:                   Early stopping threshold
"""
def train(model, train_loader, eval_loader, optimizer, criterion, device, num_epochs, writer, patience=5):
    total_loss = 0.0
    best_eval_loss = float('inf')
    epochs_no_improve = 0

    # For each epoch
    for epoch in tqdm.tqdm(range(num_epochs)):
        # Set model in training mode
        model.train()

        # Train for one epoch
        epoch_loss = trainOneEpoch(model, train_loader, device, optimizer, criterion, epoch, writer)
        writer.add_scalar("Loss per epoch - Train", epoch_loss, epoch)
        
        # Averaged loss until this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        total_loss += epoch_loss

        # Evaluate the model on eval set
        eval_loss = evaluate(model, eval_loader, criterion, device, "eval")
        writer.add_scalar("Loss/eval", eval_loss, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}, Eval Loss: {eval_loss:.4f}')

        # Early stopping
        if eval_loss < best_eval_loss:
            best_eval_loss =  eval_loss
            epochs_no_improve = 0
            # If this is the best model until this epoch,
            # save a checkpoint
            save_checkpoint(epoch, total_loss, model, optimizer)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Average loss
    average_loss = total_loss / num_epochs
    return average_loss

