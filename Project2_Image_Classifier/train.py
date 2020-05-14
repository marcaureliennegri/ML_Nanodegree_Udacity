#Importing Essential Libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

#Pytorch Libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
torch.manual_seed(42)

#Importing Image Libraries
from collections import OrderedDict
from PIL import Image


#Setting up Argparse arguments that will be passed by the user on the command line#
###################################################################################

def arg_parse():
    parser = argparse.ArgumentParser(description="This is a parser for the train.py application")
    
    #Defining main arguments
    parser.add_argument("data_dir", type=str, help="Directory containing train and validation datasets")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="Directory to save and load trained model checkpoints")
    parser.add_argument("--arch", type=str, default="densenet161", choices=["vgg16", "vgg19", "densenet121", "densenet161"], help="Pretrained model for transfer learning")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=1024, help="Number of hidden layer units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Use GPU if its available")
    
    return parser.parse_args()

#####################Defining the main function of the app#########################
###################################################################################
def main():
    #calling the Argparse arguments
    args = arg_parse()
    
    #directories containing trainning and validation sets
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    
    # Define your transforms for the training and validation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
}
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform = data_transforms['training']),
        'valid_data': datasets.ImageFolder(valid_dir, transform = data_transforms['validation'])
}
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'valid_loader':torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)
}
    
    # Load Label mapping from json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    cat_num = len(cat_to_name)
    
    # Device to train on - GPU or CPU
    device = args.device
    
    # Loading the Pretrained model chosen
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze model parameters when training; turn off gradients for our model
    for param in model.parameters():
        param.requires_grad = False
        
    # Define Classifier for pretrained model
    #Models "vgg16", "vgg19", "densenet121", "densenet161"
    if args.arch == "vgg16":
        input_num = 25088
    elif args.arch == "vgg19":
        input_num = 25088
    elif args.arch == "densenet121":
        input_num = 1024
    elif args.arch == "densenet161":
        input_num = 2208

    
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_num, args.hidden_units)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(args.hidden_units, cat_num)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    # Setting up GPU, criterion and optimezer
    #Only train the classifier parameters, feature parameters are frozen
    criterion = nn.NLLLoss()

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    # Train model
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
            
        # Training loop
        for images, labels in dataloaders['train_loader']:
            steps += 1
        
            # Move images & labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            # Set gradients to zero
            optimizer.zero_grad()
        
            # Feedforward
            logps = model(images)
            loss = criterion(logps, labels)
        
            # Backpropagation
            loss.backward()
        
            # Gradient descent
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
            
                # Turn on evaluation, inference mode; turn off dropout
                model.eval()
                valid_loss = 0
                accuracy = 0
            
                # Turn off autograd
                with torch.no_grad():

                    # Validation loop
                    for images, labels in dataloaders['valid_loader']:
                
                        # Move images and labels to GPU if available
                        images, labels = images.to(device), labels.to(device)
        
                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                print(f"Epoch {epoch+1}/{epochs}... "
                      f"Train loss: {running_loss/print_every:.4f}... "
                      f"Validation loss: {valid_loss/len(dataloaders['valid_loader']):.4f}... "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid_loader']):.4f}... ") 
            
                running_loss = 0
            
                # Set model back to training mode
                model.train()   

    # Save the checkpoint
    model.class_to_idx = image_datasets['train_data'].class_to_idx

    checkpoint = {'input_size': input_num,
                  'output_size': cat_num,
                  'pretrained_model': getattr(models, args.arch)(pretrained=True),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, args.save_dir)
    
    print("\nTraining process is completed!\n")
    print("Checkpoint was saved as: {}".format(args.save_dir))

##### Test if file is been executed as main. If False, the script sets to main()#####

if __name__ == "__main__":
    main()
