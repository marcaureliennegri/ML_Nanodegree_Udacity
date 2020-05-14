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
    parser = argparse.ArgumentParser(description="This is a parser for the predict.py application")
    #Defining main arguments
    parser.add_argument("img_path", type=str, help="Path to testing image")
    parser.add_argument("checkpoint", type=str, help="Saved trained model checkpoint")
    parser.add_argument("--top_k", type=int, default = 1, help="Top-K most probable classes. It must be an interger!")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="JSPN object mapping the integer encoded categories to the actual names of the flowers.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Setting up the Device: use GPU if available")
    
    return parser.parse_args()

def load_checkpoint(path):
    # Loading checkpoint using Pytorch
    checkpoint = torch.load(path)
    
    # Loading pretrained model
    model = checkpoint['pretrained_model']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Loading each model caracteristics form the checkpoint dictionary 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']    
    
    return model

#Defining the function: Process Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    fig = Image.open(image);
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # TODO: Process a PIL image for use in a PyTorch model
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    image = transformation(fig)
    return image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    #Using GPU if its avaiable
    args = arg_parse()
    device = args.device
    model.to(device)
    
    #changing model do evaluation mode
    model.eval()
    
    #Image processing
    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    #Loading Model   
    with torch.no_grad():
        output = model(image)
        prob = torch.exp(output)
        probs, classes = prob.topk(topk, dim=1)
        probs = probs.detach().cpu().numpy().tolist()[0]
        classes = classes.detach().cpu().numpy().tolist()[0]
    return probs, classes

def main():
    args = arg_parse()
    #Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
            
    
    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Get arguments for Class prediction
    image_path = args.img_path
    top_k = args.top_k
    
    if(args.top_k == 1):
        probs, classes = predict(image_path, model, top_k)
        names = [cat_to_name[str(idx)] for idx in classes]
        print(f"{names[0]} is the most probable class\n"
          f"Probability: {probs[0]}\n")

    else:
        probs, classes = predict(image_path, model, top_k)
        names = [cat_to_name[str(idx)] for idx in classes]
        print(f"Top {top_k} most probable classes:\n"
              f"Names: {names}\n"
              f"Probabilities: {probs}\n")
    
##### Test if file is been executed as main. If False, the script sets to main()#####
if __name__ == "__main__":
    main()
