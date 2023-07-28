import matplotlib.pyplot as plt

#Heatmaps
def show_heatmap_dim1(img_data,heatmap_data,subject,no_slices=40):
    dim = img_data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(heatmap_data[i,:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data[i,:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        #ax.set_title("Dim1 "+subject+", Slice: "+str(i))
        ax.set_title("Slice: "+str(i))
        plt.show()

def show_heatmap_dim2(img_data,heatmap_data,subject,no_slices=40):
    dim = img_data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(heatmap_data[:,i,:].T, cmap='hot',origin="lower", vmin=0, vmax=1,alpha = 1.0*(heatmap_data[:,i,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        #ax.set_title("Dim2 "+subject+", Slice: "+str(i))
        plt.show()

def show_heatmap_dim3(img_data,heatmap_data,subject,no_slices=40):
    dim = img_data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = img_data.max()
    min_val = img_data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.imshow(heatmap_data[:,:,i].T, cmap='hot',origin="lower", vmin=0, vmax=1,alpha = 1.0*(heatmap_data[:,:,i].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
        #ax.set_title("Dim3 "+subject+", Slice: "+str(i))
        plt.show()

#Show scan
def show_slices_dim1(data,no_slices=40):
    dim = data.shape[0]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    min_val = data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        #ax.set_title("Dim1, Slice: "+str(i))
        ax.set_title("Slice: "+str(i))
        plt.show()

def show_slices_dim2(data,no_slices=40):
    dim = data.shape[1]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    min_val = data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim2, Slice: "+str(i))
        plt.show()

def show_slices_dim3(data,no_slices=40):
    dim = data.shape[2]
    slice_step = int(dim/no_slices)
    if slice_step == 0:
        slice_step = 1
    max_val = data.max()
    min_val = data.min()
    for i in range(0,dim,slice_step):
        fig, ax = plt.subplots()
        ax.imshow(data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
        ax.set_title("Dim3, Slice: "+str(i))
        plt.show()


#Show img and centroids
def show_centroids_dim1(img_data,img_header,ctd_list,no_slices=40):
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
        ax.set_title("Dim1, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
        plt.show()

def show_centroids_dim2(img_data,img_header,ctd_list,no_slices=40):
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
        ax.set_title("Dim2, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
        plt.show()

def show_centroids_dim3(img_data,img_header,ctd_list,no_slices=40):
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
        ax.set_title("Dim3, Slice: "+str(i))
        for v in ctd_list[1:]:
            ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
        plt.show()

#Show img and mask
def show_mask_dim1(img_data,msk_data,no_slices=40):
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
        plt.show()

def show_mask_dim2(img_data,msk_data,no_slices=40):
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
        plt.show()

def show_mask_dim3(img_data,msk_data,no_slices=40):
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
        plt.show()


#Show mask and centroids and image
def show_everything_dim1(img_data,img_header,msk_data,ctd_list,no_slices=40):
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
            ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
        plt.show()

def show_everything_dim2(img_data,img_header,msk_data,ctd_list,no_slices=40):
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
            ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
        plt.show()

def show_everything_dim3(img_data,img_header,msk_data,ctd_list,no_slices=40):
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
            ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
        plt.show()