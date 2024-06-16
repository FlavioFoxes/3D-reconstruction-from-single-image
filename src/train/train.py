import sys
from src.utils.utils import save_checkpoint
import matplotlib.pyplot as plt
import tqdm
import matplotlib.pyplot as plt
import torch

from src.test.evaluate import *



def trainOneEpoch(model, train_loader, device, optimizer, criterion, epoch, writer):
    model.train()  # Set the model to training mode
    # DEBUG
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    loss_list = []
    running_loss = 0.0
    i = 0
    for images, point_clouds in tqdm.tqdm(train_loader):
        # DEBUG
        # Salva i pesi prima dell'ottimizzazione
        # before_update = {}
        # for name, param in model.named_parameters():
        #     before_update[name] = param.clone().detach()
        images = images.to(device)
        point_clouds = point_clouds.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # DEBUG
        # print("Images is on CUDA:", images.is_cuda)
        # print("points is on CUDA:", point_clouds.is_cuda)
        # check_model_device(model)

        outputs = model(images)  # Forward pass
        
        loss = criterion(outputs, point_clouds)  # Compute loss
        
        # Mappa l'indice normalizzato sull'asse x di TensorBoard tra 0 e total_epochs
        writer_idx = (epoch + i / 4)
        writer.add_scalar("Loss per step - Train", loss, writer_idx)
        loss.backward()  # Backward passepochs_no_improve
        
        # DEBUG
        # Stampa dei gradienti per verificare che non siano nulli
        # check_gradients(model)


        optimizer.step()  # Optimize the model

        # DEBUG
        # Confronta i pesi prima e dopo l'ottimizzazione
        # print_weight_updates(model, before_update)

        loss_list.append(float(loss))
        i += 1
        
        print(f'---Running Loss: {float(loss):.4f}')  # Print running loss


    epoch_loss = sum(loss_list) / len(loss_list)  # Compute epoch loss
    return epoch_loss

def train(model, train_loader, eval_loader, optimizer, criterion, device, num_epochs, writer, patience=5):
    total_loss = 0.0
    best_eval_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        epoch_loss = trainOneEpoch(model, train_loader, device, optimizer, criterion, epoch, writer)
        writer.add_scalar("Loss per epoch - Train", epoch_loss, epoch)
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
            save_checkpoint(epoch, total_loss, model, optimizer)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch}")
                break


    average_loss = total_loss / num_epochs
    return average_loss

