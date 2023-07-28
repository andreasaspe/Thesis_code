import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
import nibabel

def Predict(dataloader, model, checkpoint_dir, output_dir):
    """
    Predicts data and saves it

    Arguments:
    dataloader - a Pytorch dataloader with data to predict
    model - the architecture of the network network defined as a pytorch model. For instance UNet3D()
    checkpoint_dir - the directory for the file containing the checkpoint of the model
    output_dir - the directory for where to save the output. Will create the folder if it does not exits.
    """
    
    if not os.path.exists(output_dir):
               os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)

    #Define model
    model = model
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode
    model.eval()    

    for inputs, subject in tqdm(dataloader):
        assert len(subject) == 1 #Make sure we are inly prediction one batch
        predictions = model(inputs)
        predictions = predictions.squeeze()
        subjectname_without_img = subject[0][:-4]
        torch.save(predictions, os.path.join(output_dir,subjectname_without_img + '_heatmap_pred.pt'))

def Plot_loss(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    plt.figure()
    plt.plot(train_loss, label='Training',linewidth=1)
    plt.plot(val_loss, label='Validation',linewidth=1)
    plt.title(r'Training-curve')
    plt.xlabel(r'Epochs')
    plt.ylabel(r'MSE-loss') #r'$\mathcal{L} ( \mathbf{x} )$'
    plt.legend(loc='upper right')
    plt.show()

# def Plot_predictions(loss_fn,dir_img,dir_heatmap):
#     all_subjects = []
#     for filename in listdir(dir_img):
#         subject = filename.split("_")[0]
#         all_subjects.append(subject)

#     for subject in all_subjects:
#             filename_img = [f for f in listdir(dir_img) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
#             img_nib = nib.load(os.path.join(dir_img,filename_img))
