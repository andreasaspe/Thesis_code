import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
import numpy as np
from skimage.measure import label, regionprops
import os
from matplotlib.patches import Circle
import copy
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors




    #Indsæt dette hvis det fucker: plt.style.use('default')

v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}


plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12  # Default text - så fx. akse-tal osv.
plt.rcParams["axes.titlesize"] = 12  # Size for titles
plt.rcParams["axes.labelsize"] = 15  # Size for labels
plt.rcParams["legend.fontsize"] = 12  # Size for legends
plt.rcParams["figure.figsize"] = (6.4, 4.8) #Standard. Målt i inches. Width x height

# plt.style.use('default')
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 12  # Default text - så fx. akse-tal osv.
# plt.rcParams["axes.titlesize"] = 12  # Size for titles
# plt.rcParams["axes.labelsize"] = 15  # Size for labels
# plt.rcParams["legend.fontsize"] = 12  # Size for legends
# plt.rcParams["figure.figsize"] = (6.4, 4.8) #Standard. Målt i inches. Width x height

#Heatmaps
#With image
def show_heatmap_img_dim1(img_data,heatmap_data,subject,alpha=0.4,no_slices=40):
    plt.style.use('default')
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    # heatmap_data = np.where(heatmap_data > 0.3,1,0)
    for i in range(0,dim,slice_step):
        #if i == 240:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin=min_val,vmax=max_val)
        ax.imshow(heatmap_data[i,:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data[i,:,:].T>alpha)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        # ax.imshow(heatmap_data[i,:,:].T, origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data[i,:,:].T>alpha)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.tight_layout()
        plt.axis('off')
        # plt.savefig('/Users/andreasaspe/Desktop/heatmap',dpi=500)
        plt.show()

def show_heatmap_img_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin=min_val,vmax=max_val)
        ax.imshow(heatmap_data[:,i,:].T, cmap='hot',origin="lower", vmin=0, vmax=1,alpha = 1.0*(heatmap_data[:,i,:].T>alpha)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.show()

def show_heatmap_img_dim3(img_data,heatmap_data,subject,alpha=0.3,no_slices=40):
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin=min_val,vmax=max_val)
        ax.imshow(heatmap_data[:,:,i].T, cmap='hot',origin="lower", vmin=0, vmax=1,alpha = 1.0*(heatmap_data[:,:,i].T>alpha)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.show()
        
#Without image
def show_heatmap_dim1(heatmap_data,subject,alpha=0.4,no_slices=40):
    plt.style.use('default')
    dim = heatmap_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        #i=48
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data[i,:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        plt.tight_layout()
        plt.axis('off')
        # Add colorbar
        #cbar = fig.colorbar(im, ax=ax)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        # if i == 50:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig('Heatmap', transparent=True))
        plt.show()
        #break

def show_heatmap_dim2(heatmap_data,subject,alpha=0.4,no_slices=40):
    dim = heatmap_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(heatmap_data[:,i,:].T, cmap='hot',origin="lower", vmin=0, vmax=1) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.show()

def show_heatmap_dim3(heatmap_data,subject,alpha=0.4,no_slices=40):
    dim = heatmap_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(heatmap_data[:,:,i].T, cmap='hot',origin="lower", vmin=0, vmax=1) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.show()
        

#Show one slice
def show_one_slice(data,subject,convert=0,no_slices=40):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = data.max()
    min_val = data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    # colors = ['black', 'blue', 'purple'] #Tror måske ikke jeg behøver at tilføje den her også?
    # cmap = plt.cm.colors.ListedColormap(colors) 
    im = ax.imshow(data.T,cmap='gray',origin="lower",vmin = min_val, vmax = max_val) #evt vmin = -1000, vmax = 2000
    # plt.axis('off')
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(40)  
    plt.show()
    
def show_one_mask_img_slice(img_data,msk_data,subject,convert=0,no_slices=40):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    # colors = ['black', 'blue', 'purple'] #Tror måske ikke jeg behøver at tilføje den her også?
    # cmap = plt.cm.colors.ListedColormap(colors) 
    color_html = "#c37feb"
    rgba_color = mcolors.to_rgba(color_html, alpha=0.4)
    colors = ['black', rgba_color] #Tror måske ikke jeg behøver at tilføje den her også?
    cmap = plt.cm.colors.ListedColormap(colors) 


    im = ax.imshow(img_data.T,cmap='gray',origin="lower",vmin = min_val, vmax = max_val) #evt vmin = -1000, vmax = 2000
    ax.imshow(msk_data.T, cmap=cmap,origin="lower",alpha =1.0*(msk_data.T>0)) #        ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",alpha =1.0*(msk_data[i,:,:].T>0))
    # plt.axis('off')
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(40)  
    plt.show()
    
    
def show_one_slice_with_pixel_markings(data,pixels,color,subject,no_slices=40):
    """
    This function can plot the extremities used for fracture measuring
    
    data: 2D data
    pixels: Tuple of coordinates (x1,y1,x2,y2)
    color: Farve på pixels
    subject: ID for patient
    """
    
    #Unpack coordinates
    x1,y1,x2,y2 = pixels
    
    data = copy.copy(data)
    
    #Mark pixel as twos
    data[x1,y1] = 2
    data[x2,y2] = 2
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define a custom colormap with black, white, and red
    colors = ['black', 'white', color] #Tror måske ikke jeg behøver at tilføje den her også?
    cmap = plt.cm.colors.ListedColormap(colors) 
    
    # Plot the image using the custom colormap
    plt.imshow(data.T, cmap=cmap,origin="lower")
    
    
    # Draw a line between the two pixels
    plt.plot([x1,x2],[y1,y2], color=color, linewidth=1)
    # plt.xlim()
    
    # Set the colorbar to show the colormap values
    # plt.colorbar(ticks=[0, 1, 2], label='Color')
    # plt.axis('off')
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(5)  

    # Show the plot
    plt.show()
    
def show_one_slice_with_two_markings(data,min_pixel,max_pixel,color_min,color_max,subject,no_slices=40):
    """
    This function can plot the extremities used for fracture measuring for TWO pixels. Compare min and max!
    
    data: 2D data
    min_pixel: Tuple of minimum coordinates (x1,y1,x2,y2)
    max_pixel: Tuple of maximum coordinates (x1,y1,x2,y2)
    color_min: Farve på min_pixel
    color_max: Farve på max_pixel
    subject: ID for patient
    """
    
    #Unpack coordinates
    x1_min,y1_min,x2_min,y2_min = min_pixel
    x1_max,y1_max,x2_max,y2_max = max_pixel
    
    data = copy.copy(data)
    
    #Mark min pixels as twos
    data[x1_min,y1_min] = 2
    data[x2_min,y2_min] = 2
    
    #Mark max pixels as threes
    data[x1_max,y1_max] = 3
    data[x2_max,y2_max] = 3
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define a custom colormap with black, white, and red
    colors = ['black', 'white', color_min, color_max] #Tror måske ikke jeg behøver at tilføje dem her også?
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Plot the image using the custom colormap
    plt.imshow(data.T, cmap=cmap,origin="lower")
    
    # Draw a line between the two pixels
    plt.plot([x1_min,x2_min],[y1_min,y2_min], color=color_min, linewidth=5)
    plt.plot([x1_max,x2_max],[y1_max,y2_max], color=color_max, linewidth=5)

    # plt.xlim()
    
    # Set the colorbar to show the colormap values
    # plt.colorbar(ticks=[0, 1, 2], label='Color')
    # plt.axis('off')
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(40)  

    # Show the plot
    plt.show()
    

#Show scan
def show_slices_dim1(data,subject,convert=0,zooms=None,no_slices=40):
    dim = data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    min_val = data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/y #y-akse over x-akse
    else:
        ratio = 1
    for i in range(0,dim,slice_step):
        #if i == 240:
        #i=48
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val) #evt vmin = -1000, vmax = 2000
        # ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.tight_layout()
        plt.axis('off')
        ax.set_aspect(ratio)
        # Add colorbar
        #cbar = fig.colorbar(im, ax=ax)
        #plt.axis('off')
        # plt.savefig('/Users/andreasaspe/Desktop/normal',dpi=500)
        plt.show()
        #break


        # fig, ax = plt.subplots()
        # ax.grid(False)
        # ax.imshow(data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        # #plt.axis('off')
        # #ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        # # if i == 50:
        # #     #figsize 8,8
        # #     plt.tight_layout()
        # #     plt.axis('off')
        # #     plt.savefig('Scan')
        # plt.show()

def show_slices_dim2(data,subject,convert=0,zooms=None,no_slices=40):
    dim = data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    min_val = data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/x #y-akse over x-akse
    else:
        ratio = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.imshow(data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        ax.set_aspect(ratio)
        plt.show()

def show_slices_dim3(data,subject,convert=0,zooms=None,no_slices=40):
    dim = data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    if convert == 1:
        max_val = 1000
        min_val = -200
    min_val = data.min()
    if zooms is not None:
        x,y,z = zooms
        ratio = y/x #y-akse over x-akse
    else:
        ratio = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.imshow(data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        ax.set_aspect(ratio)
        plt.show()


#Show img and centroids in the verse dataformat
def show_centroids_dim1(img_data,ctd_list,subject,text=0,markersize=10,no_slices=40):
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[2],v[3]),markersize))
            if text==1:
                plt.text(v[2], v[3], v_dict[v[0]], color='red', fontsize=12)
        plt.show()

def show_centroids_dim2(img_data,ctd_list,subject,text=0,markersize=10,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[3]), markersize))
            if text==1:
                plt.text(v[1], v[3], v_dict[v[0]], color='red', fontsize=12)
        plt.show()

def show_centroids_dim3(img_data,ctd_list,subject,text=0,markersize=10,no_slices=40):
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[2]), markersize))
            if text==1:
                plt.text(v[1], v[2], v_dict[v[0]], color='red', fontsize=12)
        plt.show()

#Show img and centroids without the first label dimension
def show_centroids_new_dim1(img_data,ctd_list,subject,markersize=1.5,no_slices=40):
    """
    Denne funktion er lavet til ctd_list i mit eget format. Og IKKE til Verse format. Det er eneste forskel.

    """
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list:
            ax.add_patch(Circle((v[1],v[2]), markersize,facecolor='red'))
        plt.show()
        
def show_centroids_new_dim2(img_data,size,ctd_list,subject,markersize=1.5,no_slices=40):
    """
    Denne funktion er lavet til ctd_list i mit eget format. Og IKKE til Verse format. Det er eneste forskel.

    """
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        #ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list:
            ax.add_patch(Circle((v[0],v[2]), markersize))
        plt.show()

def show_centroids_new_dim3(img_data,size,ctd_list,subject,markersize=1.5,no_slices=40):
    """
    Denne funktion er lavet til ctd_list i mit eget format. Og IKKE til Verse format. Det er eneste forskel.

    """
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        #ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list:
            ax.add_patch(Circle((v[0],v[1]), markersize))
        plt.show()
        
        
#Compare centroids in same plot
def show_centroids_comparison_dim1(img_data,ctd_list_GT,ctd_list_pred,subject,text=1,markersize=10,no_slices=40):
    """
    Compare GT and prediction. ctd_list should be verseformat 
    
    """
    dim = img_data.shape[0]
    dim2 = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots(figsize=(dim+100,dim2))
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list_GT[1:]:
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='cyan'))
            if text==1:
                plt.text(v[2]-20, v[3], v_dict[v[0]], color='cyan', fontsize=12)
        for v in ctd_list_pred[1:]:
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='red'))
            if text==1:
                plt.text(v[2]+10, v[3], v_dict[v[0]], color='red', fontsize=12)
        plt.show()
        
#Compare centroids to report
def show_centroids_comparison_toreport_dim1(img_data,ctd_list_GT,ctd_list_pred,subject,text=0,markersize=10,no_slices=40):
    """
    Compare GT and prediction. ctd_list should be verseformat 
    
    """
    dim = img_data.shape[0]
    dim2 = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots() #figsize=(dim/14,dim2/14)
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        # ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        ax.set_title("Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list_GT[1:]:
            # ax.plot(v[2],v[3], c='cyan', marker='X',linewidth=2,markersize=10)
            # ax.scatter(v[2],v[3], s=markersize, c='cyan', marker='X')
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='blue'))
            if text==1:
                #Without box
                plt.text(v[2]+10, v[3]-3, v_dict[v[0]], color='white', weight='bold')
                #With box
                # plt.text(v[2]+10, v[3], v_dict[v[0]], color='black', fontsize=10, weight='bold', # Set the background color to blue
             # bbox=dict(facecolor='white',alpha=0.9, edgecolor='black',boxstyle='round,pad=0.1'))
        for v in ctd_list_pred[1:]:
            # ax.scatter(v[2],v[3], s=markersize, c='red', marker='X')
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='red'))
            # Manually create the legend
            
            
        # Create custom proxy artists for the legend
        target_legend = Line2D([0], [0], label='Target', marker='o', color='blue', markersize=markersize+1, linestyle='')
        prediction_legend = Line2D([0], [0], label='Prediction', marker='o', color='red', markersize=markersize+1, linestyle='') #Før var den markersize=5
        
        # target_legend = mpatches.Patch(color='cyan', label='Target')
        # prediction_legend = mpatches.Patch(color='red', label='Prediction')
    
        # Show the legend with custom proxy artists
        #Det her er den rigtige i results
        # ax.legend(handles=[target_legend, prediction_legend], markerscale=1.5, framealpha=0.8,loc='upper right', borderpad = 0.2, fontsize=12) #loc='upper right',bbox_to_anchor=(0,1), borderaxespad=0.
        #Det her er til discussion!
        ax.legend(handles=[target_legend, prediction_legend], markerscale=1.5, framealpha=0.8,loc='upper right', borderpad = 0.1, fontsize=12) 
        
        # ax.legend(['Target', 'Prediction'],marker=['x','x'],color=['red','cyan'])
        plt.show()
        
#Compare centroids to report RH
def show_centroids_RH_toreport_dim1(img_data,ctd_list,subject,text=0,markersize=10,no_slices=40):
    """
    Compare GT and prediction. ctd_list should be verseformat 
    
    """
    dim = img_data.shape[0]
    dim2 = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots() #figsize=(dim/14,dim2/14)
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        # ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        ax.set_title("Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list[1:]:
            # ax.scatter(v[2],v[3], s=markersize, c='red', marker='X')
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='red'))
            if text==1:
                #Without box
                plt.text(v[2]+10, v[3]-3, v_dict[v[0]], color='white', weight='bold')
            
            
        # Create custom proxy artists for the legend
        # target_legend = Line2D([0], [0], label='Target', marker='o', color='blue', markersize=markersize+1, linestyle='')
        # prediction_legend = Line2D([0], [0], label='Prediction', marker='o', color='red', markersize=markersize+1, linestyle='') #Før var den markersize=5
        
        # ax.legend(['Target', 'Prediction'],marker=['x','x'],color=['red','cyan'])
        plt.show()


#Compare centroids to report RH
def show_centroids_RH_toreport_dim2(img_data,ctd_list,subject,text=0,markersize=10,no_slices=40):
    """
    Compare GT and prediction. ctd_list should be verseformat 
    
    """
    dim = img_data.shape[0]
    dim2 = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots() #figsize=(dim/14,dim2/14)
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        # ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        ax.set_title("Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list[1:]:
            # ax.scatter(v[2],v[3], s=markersize, c='red', marker='X')
            ax.add_patch(Circle((v[1],v[3]), markersize,facecolor='red'))
            if text==1:
                #Without box
                plt.text(v[1]+10, v[3]-3, v_dict[v[0]], color='white', weight='bold')
            
            
        # Create custom proxy artists for the legend
        # target_legend = Line2D([0], [0], label='Target', marker='o', color='blue', markersize=markersize+1, linestyle='')
        # prediction_legend = Line2D([0], [0], label='Prediction', marker='o', color='red', markersize=markersize+1, linestyle='') #Før var den markersize=5
        
        # ax.legend(['Target', 'Prediction'],marker=['x','x'],color=['red','cyan'])
        plt.show()
                
#Compare centroids in same plot
def show_centroids_mapping_dim1(img_data,ctd_list_GT,ctd_list_pred,subject,text=1,markersize=10,no_slices=40):
    """
    Compare GT and prediction. Numbering is 1+2+3+4 and so on. Format for ctd_list_pred is my own format.
    
    """
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        for v in ctd_list_GT[1:]:
            ax.add_patch(Circle((v[2],v[3]), markersize,facecolor='cyan'))
            if text==1:
                plt.text(v[2]-20, v[3], v_dict[v[0]], color='cyan', fontsize=12)
        for i, v in enumerate(ctd_list_pred):
            ax.add_patch(Circle((v[1],v[2]), markersize,facecolor='red'))
            if text==1:
                plt.text(v[1]+10, v[2], str(i+1), color='red', fontsize=12)
        plt.show()
    
        

def show_COM_dim1(img_data,COM,subject,markersize=10,no_slices=40):
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        # plt.axis('off')
        ax.add_patch(Circle((COM[1],COM[2]), markersize))
        plt.show()

def show_COM_dim2(img_data,size,COM,subject,markersize=10,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        # plt.axis('off')
        ax.add_patch(Circle((COM[0],COM[2]), markersize))
        plt.show()
        
def show_COM_dim3(img_data,size,COM,subject,markersize=10,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        # plt.axis('off')
        ax.add_patch(Circle((COM[0],COM[1]), markersize))
        plt.show()
        

#Show img and mask
def show_mask_img_dim1(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[i,:,:].T,origin="lower",alpha =0.4*(msk_data[i,:,:].T>0)) #        ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",alpha =1.0*(msk_data[i,:,:].T>0))
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_img_dim2(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,i,:].T,origin="lower",alpha =0.5*(msk_data[:,i,:].T>0))
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_img_dim3(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,:,i].T,origin="lower",alpha =0.5*(msk_data[:,:,i].T>0))
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()
        
        
        
#Show img and mask with range filtered to L5 to T10
def show_mask_img_filtered_dim1(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower", vmin = 17, vmax = 24, alpha =1.0*(msk_data[i,:,:].T>0))       
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_img_filtered_dim2(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,i,:].T,cmap="jet",origin="lower", vmin = 17, vmax = 24, alpha =1.0*(msk_data[:,i,:].T>0))       
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_img_filtered_dim3(img_data,msk_data,subject,no_slices=40):
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower", vmin = 17, vmax = 24, alpha =1.0*(msk_data[:,:,i].T>0))       
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()
        
#Show mask
def show_mask_dim1(msk_data,subject,no_slices=40):
    dim = msk_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(msk_data[i,:,:].T,origin="lower",cmap="gray",vmin =0, vmax = 1)
        #ax.set_facecolor('black')
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_dim2(msk_data,subject,no_slices=40):
    dim = msk_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(msk_data[:,i,:].T,origin="lower",cmap="gray",vmin =0, vmax = 1)
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_dim3(msk_data,subject,no_slices=40):
    dim = msk_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(msk_data[:,:,i].T,origin="lower",cmap="gray",vmin =0, vmax = 1)
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()
        
#Show mask with range filtered to L5 to T10
def show_mask_filtered_dim1(msk_data,subject,no_slices=40):
    dim = msk_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower", vmin = 0, vmax = 24, alpha = 0.5*(msk_data[i,:,:].T>0))   
        #ax.set_facecolor('black')
        ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_filtered_dim2(msk_data,subject,no_slices=40):
    dim = msk_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(msk_data[:,i,:].T,cmap="jet",origin="lower", vmin = 0, vmax = 24)   
        ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()

def show_mask_filtered_dim3(msk_data,subject,no_slices=40):
    dim = msk_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower", vmin = 0, vmax = 24)   
        ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.axis('off')
        plt.show()
        
def show_boundingbox_dim1(img_data,coordinates,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates: The coordinates of the bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
    
    #Get image dimension
    _, dim2, dim3 = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary. 
    min_slack2 = 0
    max_slack2 = 0
    min_slack3 = 0
    max_slack3 = 0
    if y_min == 0:
        min_slack2 = 0.01
    if y_max == dim2:
        max_slack2 = 0.01
    if z_min == 0:
        min_slack3 = 0.01
    if z_max == dim3:
        max_slack3 = 0.01
    
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((y_min, z_min), (y_max-y_min), (z_max-z_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    slice_step = int((int(x_max) - int(x_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(x_min),int(x_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect)) #??? Tilføjede en ekstra copy nu. Men har jo virket før? Det virkede i hvert fald nu.
        ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack2*dim2,dim2 + min_slack2*dim2) #Add slack
        ax.set_ylim(0 - min_slack3*dim3,dim3 + max_slack3*dim3) #Add slack
        plt.axis('off')
        ax.set_aspect(ratio)
        # if i == 202:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim1"))
        plt.show()
    
    
    
def show_boundingbox_dim2(img_data,coordinates,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates: The coordinates of the bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
          
    #Get image dimension
    dim1, _, dim3 = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary.
    min_slack1 = 0
    max_slack1 = 0
    min_slack3 = 0
    max_slack3 = 0
    if x_min == 0:
        min_slack1 = 0.01
    if x_max == dim1:
        max_slack1 = 0.01
    if z_min == 0:
        min_slack3 = 0.01
    if z_max == dim3:
        max_slack3 = 0.01
        
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, z_min), (x_max-x_min), (z_max-z_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    slice_step = int((int(y_max) - int(y_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(y_min),int(y_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect))
        #ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack1*dim1,dim1 + min_slack1*dim1) #Add slack
        ax.set_ylim(0 - min_slack3*dim3,dim3 + max_slack3*dim3) #Add slack
        ax.set_aspect(ratio)
        #plt.axis('off')
        # if i == 136:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim2"))
        plt.show()
    
    
def show_boundingbox_dim3(img_data,coordinates,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates: The coordinates of the bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates

    #Get image dimension
    dim1, dim2, _ = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary.
    min_slack1 = 0
    max_slack1 = 0
    min_slack2 = 0
    max_slack2 = 0
    if y_min == 0:
        min_slack2 = 0.01
    if y_max == dim2:
        max_slack2 = 0.01
    if z_min == 0:
        min_slack2 = 0.01
    if z_max == dim2:
        max_slack2 = 0.01
        
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = y/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    slice_step = int((int(z_max) - int(z_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(z_min),int(z_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect))
        #ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack1*dim1,dim1 + min_slack1*dim1) #Add slack
        ax.set_ylim(0 - min_slack2*dim2,dim2 + min_slack2*dim2) #Add slack
        ax.set_aspect(ratio)
        #plt.axis('off')
        # if i == 128:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim3"))
        plt.show()
    
    


def show_boundingboxes_dim1(img_data,coordinates1,coordinates2,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates1: The coordinates of the first bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # coordinates2: The coordinates of the second bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates1
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = coordinates2
    
    #Get image dimension
    _, dim2, dim3 = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary. 
    min_slack2 = 0
    max_slack2 = 0
    min_slack3 = 0
    max_slack3 = 0
    if y_min == 0:
        min_slack2 = 0.01
    if y_max == dim2:
        max_slack2 = 0.01
    if z_min == 0:
        min_slack3 = 0.01
    if z_max == dim3:
        max_slack3 = 0.01
    
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((y_min, z_min), (y_max-y_min), (z_max-z_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((y_min2, z_min2), (y_max2-y_min2), (z_max2-z_min2), linewidth=linewidth, edgecolor='b', facecolor='none')
    slice_step = int((int(x_max) - int(x_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(x_min),int(x_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect2))
        ax.add_patch(copy.copy(rect))
        ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack2*dim2,dim2 + min_slack2*dim2) #Add slack
        ax.set_ylim(0 - min_slack3*dim3,dim3 + max_slack3*dim3) #Add slack
        plt.axis('off')
        ax.set_aspect(ratio)
        
        # Add the legend
        legend_handles = [rect2, rect]
        legend_labels = ["Target", "Prediction"]
        ax.legend(legend_handles, legend_labels, framealpha=0.8, fontsize = 16, borderpad = 0.2) #loc='lower right'
        
        # if i == 202:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim1"))
        plt.show()
        


def show_boundingboxes_dim2(img_data,coordinates,coordinates2,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates: The coordinates of the bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = coordinates2

          
    #Get image dimension
    dim1, _, dim3 = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary.
    min_slack1 = 0
    max_slack1 = 0
    min_slack3 = 0
    max_slack3 = 0
    if x_min == 0:
        min_slack1 = 0.01
    if x_max == dim1:
        max_slack1 = 0.01
    if z_min == 0:
        min_slack3 = 0.01
    if z_max == dim3:
        max_slack3 = 0.01
        
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = z/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, z_min), (x_max-x_min), (z_max-z_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((x_min2, z_min2), (x_max2-x_min2), (z_max2-z_min2), linewidth=linewidth, edgecolor='b', facecolor='none')
    slice_step = int((int(y_max) - int(y_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(y_min),int(y_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect2))
        ax.add_patch(copy.copy(rect))
        #ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack1*dim1,dim1 + min_slack1*dim1) #Add slack
        ax.set_ylim(0 - min_slack3*dim3,dim3 + max_slack3*dim3) #Add slack
        ax.set_aspect(ratio)
        #plt.axis('off')
        # if i == 136:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim2"))
        plt.show()
    

def show_boundingboxes_dim3(img_data,coordinates,coordinates2,subject,convert=0,zooms = None,linewidth=4,no_slices=40):
    # This function can plot a bounding box on top of image
    #
    # Inputs
    # img_data:    The image data as array
    # coordinates: The coordinates of the bounding box as a tuple. This is the output of the function BoundingBox
    #              The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    # subject:     A string containing the name of the subject.

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = coordinates2


    #Get image dimension
    dim1, dim2, _ = img_data.shape
    #Check if bounding box is at the image border. If so, we want to add slack to the ax limits in order to show the box at the boundary.
    min_slack1 = 0
    max_slack1 = 0
    min_slack2 = 0
    max_slack2 = 0
    if y_min == 0:
        min_slack2 = 0.01
    if y_max == dim2:
        max_slack2 = 0.01
    if z_min == 0:
        min_slack2 = 0.01
    if z_max == dim2:
        max_slack2 = 0.01
        
    #PLOT
    max_val = img_data.max()
    min_val = img_data.min()
    if convert == 1:
        max_val = 1000
        min_val = -200
    if zooms is not None:
        x,y,z = zooms
        ratio = y/x #y-akse over x-akse
    else:
        ratio = 1
    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=linewidth, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((x_min2, y_min2), (x_max2-x_min2), (y_max2-y_min2), linewidth=linewidth, edgecolor='b', facecolor='none')
    slice_step = int((int(z_max) - int(z_min))/no_slices)
    if slice_step == 0:
        slice_step = 1
    for i in range(int(z_min),int(z_max),slice_step):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.add_patch(copy.copy(rect2))
        ax.add_patch(copy.copy(rect))
        #ax.set_title(subject+" Slice "+str(i))
        ax.set_facecolor('black') #Make data outside of range black instead of white
        ax.set_xlim(0 - min_slack1*dim1,dim1 + min_slack1*dim1) #Add slack
        ax.set_ylim(0 - min_slack2*dim2,dim2 + min_slack2*dim2) #Add slack
        ax.set_aspect(ratio)
        #plt.axis('off')
        # if i == 128:
        #     #figsize 8,8
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.savefig(os.path.join("E:/Andreas_s174197/Thesis/Figures/RH_bounding_box",subject+"_slice"+str(i)+"_dim3"))
        plt.show()
    
    
    

#Show mask and centroids and image
def show_everything_dim1(img_data,img_header,msk_data,ctd_list,markersize=10,no_slices=40):
    zooms = img_header.get_zooms()
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[i,:,:].T>0))
        ax.set_title("Dim1, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[2],v[3]), markersize))
        plt.show()

def show_everything_dim2(img_data,img_header,msk_data,ctd_list,markersize=10,no_slices=40):
    zooms = img_header.get_zooms()
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[:,i,:].T>0))
        ax.set_title("Dim2, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[3]), markersize))
        plt.show()

def show_everything_dim3(img_data,img_header,msk_data,ctd_list,markersize=10,no_slices=40):
    zooms = img_header.get_zooms()
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[:,:,i].T>0))
        ax.set_title("Dim3, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[2]), markersize))
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
    #SKRALD
        
        # label_im = label(volume)

        # max_val = img_data.max()
        # min_val = img_data.min()
        # # Create a Rectangle patch
        # #rect = patches.Rectangle((y_min, z_min), (y_max-y_min), (z_max-z_min), linewidth=1, edgecolor='r', facecolor='none')
        # for i in range(0,dim1):
        #     fig, ax = plt.subplots()
        #     ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        #     #ax.add_patch(copy(rect))
        #     for i in regionprops(label_im):
        #         _,_, maxr, maxc,minr, minc, = i.bbox
        #         rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                                   fill=False, edgecolor='red', linewidth=2)
        #         ax.add_patch(rect)
        #         ax.set_axis_off()
        #     plt.show()
            

        # max_val = img_data.max()
        # min_val = img_data.min()
        # # Create a Rectangle patch
        # rect = patches.Rectangle((y_min, z_min), (y_max-y_min), (z_max-z_min), linewidth=1, edgecolor='r', facecolor='none')
        # for i in range(x_min,x_max):
        #     fig, ax = plt.subplots()
        #     ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        #     ax.add_patch(copy(rect))
        #     ax.set_title(subject)
        #     plt.show()



