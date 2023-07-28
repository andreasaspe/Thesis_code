class CustomDataset(Dataset):
    def __init__(self,img_path,label_path,with_label_closing = False):
        self.label_path = label_path
        self.img_path = img_path
        file_list_img = sorted(glob(self.img_path + "/*"))
        file_list_lab = sorted(glob(self.label_path + "/*"))
        self.imgs = []
        self.labels = []
        self.closing = with_label_closing
        for image in file_list_img:
            self.imgs.append(image)
        for label in file_list_lab:
            self.labels.append(label)
#         nr = len(file_list_lab)
#         for i in range(nr):
#             lab = np.load(file_list_lab[i])
#             n,m = np.shape(lab)
#             if np.sum(lab)>=0.1*n*m:
#                 self.labels.append(file_list_lab[i])
#                 self.imgs.append(file_list_img[i])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.imgs[idx],self.labels[idx]
        img = np.load(img_path)
        lab = np.load(label_path)
        if self.closing == True:
            kernel = np.ones((5,5),np.uint8)
#             lab = cv2.dilate(lab, kernel, iterations=1)
            lab = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, kernel)
        return img,lab
    
    def show_slice_and_label(self,idx):
        img_path, label_path = self.imgs[idx],self.labels[idx]
        one_slice = np.load(img_path)
        labels = np.load(label_path)
        if self.closing == True:
            kernel = np.ones((5,5),np.uint8)
#             labels = cv2.dilate(labels, kernel, iterations=1)
            labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(one_slice,cmap='gray')
        ax1.title.set_text('Image')
        ax2.imshow(labels,cmap = 'gray')
        ax2.title.set_text('Labels')