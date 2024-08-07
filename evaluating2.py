import datasets
from learning import normalize, DEVICE
from encoders import EncoderGMM
import timm
import torch
import os
import numpy as np
import pickle as pk
import itertools
from itertools import zip_longest
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
from skimage import io, transform
from PIL import Image
from torchvision import models, transforms
from os.path import normpath, basename
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from skimage import feature
import cv2
from typing import List, Tuple, Union
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier




def add_noise(x):
    return x

def classify(dataname, modelname, resolution, kernels, layers, method, classifiername, path, snr:float=None):

    ds_data,classes = datasets.load(dataname, size=resolution, path=path, bsize=1, shuffle=False, snr=snr)
    model_fc = timm.create_model(modelname, pretrained=True, num_classes=0).eval().to(DEVICE)
    model_fv = timm.create_model(modelname, pretrained=True, features_only=True).eval().to(DEVICE)
    torch.set_grad_enabled(False)

    layers_sizes = []
    x = torch.randn((1,3,112,112)).to(DEVICE)
    o = model_fv(x)
    del x
    for y in o:
        layers_sizes.append(y.shape[1])
    outsize = layers_sizes[layers[0]]
        
    if method == 'pca':
        def reduce_dim(x, mdl):
            x = x.numpy()
            s = x.shape
            x = x.reshape(-1,s[2])
            return mdl(x).reshape(s[0],-1,outsize)
    else:
        def reduce_dim(x, mdl):
            return mdl(x).numpy()

    if snr is None:
        print("Evaluating model")
    else:
        print(f"Evaluating model with SNR {snr}")
   
    n = len(ds_data)
    for i in range(n):
        enc = EncoderGMM(kernels)
        gmmfile = os.path.join(path,f"gmm/train_{i}.pkl")
        enc.gmm = pk.load(open(gmmfile,'rb'))
        enc.fitted = True

        mdls = {layers[0]:lambda x: x}
        pcas = {}
        for l in layers[1:]:
            if method == 'pca':
                pcafile = os.path.join(path,f"pca/{i}_{l}.pkl")
                pcas[l] = pk.load(open(pcafile,'rb'))
                mdls[l] = lambda x: pcas[l].transform(x)
                continue
            if method == 'ae':
                pcafile = os.path.join(path,f"ae/{i}_{l}.pkl")
                pcas[l] = pk.load(open(pcafile,'rb'))
                pcas[l].to('cpu')
                mdls[l] = lambda x: pcas[l].encoder(x)
                continue
            insize = layers_sizes[l]
            stride = insize//outsize
            kernel = insize-stride*outsize+1
            if method == 'max':
                mdls[l] = torch.nn.MaxPool2d((1,kernel),(1,stride),0)
            else:
                mdls[l] = torch.nn.AvgPool2d((1,kernel),(1,stride),0)

        clf_fv = pk.load(open(os.path.join(path,f"classifier/{classifiername}_fv_{i}.pkl"),'rb'))
        clf_fcfv = pk.load(open(os.path.join(path,f"classifier/{classifiername}_fcfv_{i}.pkl"),'rb'))

        total = len(ds_data[i]['val'].dataset)
        y_true = np.zeros(total)
        y_fv = np.zeros(total)
        y_fcfv = np.zeros(total)
        count = 0
        
        #vec_mat = np.zeros((len(classes), n))
        features = []
        
        for inputs,labels in ds_data[i]['val']:
            count+= inputs.shape[0]
            print("Round {}/{} - {:05d}/{:05d}".format(i+1,n,count,total),end="\r")
            inputs = add_noise(inputs).to(DEVICE)
            x_fc = model_fc(inputs).to('cpu').numpy()
            o = model_fv(inputs)
            
            x_fv = None
            for l in layers:
                s = o[l].shape
                x = o[l].to('cpu').reshape(s[0],s[1],-1).swapaxes(1,2)
                x = reduce_dim(x,mdls[l])
                if x_fv is None:
                    x_fv = x
                else:
                    x_fv = np.concatenate((x_fv,x),axis=1)

            x_fc = normalize(x_fc)
            x_fv = normalize(enc.transform(x_fv))
            x = np.concatenate((x_fc,x_fv),axis=1)

            y_fv[count-1] = clf_fv.predict(x_fv)[0]
            y_fcfv[count-1] = clf_fcfv.predict(x)[0]
            y_true[count-1] = labels.numpy()[0]
            
            features.append(x.flatten())

        layer = ''.join([str(l) for l in layers])
        endfile = '.npz' if snr is None else f'_{snr}.npz'
        np.savez_compressed(os.path.join(path,f"results/{dataname}_{modelname}_{kernels}_{resolution}_{layer}_{method}_{i}{endfile}"),
                            y_pred1 = y_fv, 
                            y_pred2 = y_fcfv,
                            y_true = y_true)

    print("{:<100}".format("Done"))
    
# -----------------------------------------------------------------------------------------------
# CLASSE QUE PERSONALIZA O DATASET
# -----------------------------------------------------------------------------------------------

    d = 0
    avg_epoch_loss = []
    avg_train_loss = []
    avg_best_acc = []
    avg_mat = []
    avg_alphas = []
    avg_nb = []
    avg_dt = []
    avg_clf1 = []
    avg_clf2 = []
    avg_clf3 = []
    avg_clf4 = []
    avg_lbp = []
    avg_lbpnn = []
    avg_lbpnn2 = []
    avg_ens = []
    avg_recall = []
    avg_time = []
    avg_prec_rec = []


    seed1 = 279

    class TextureDataset(Dataset):

        def __init__(self, root_dir, sample, transform=None):

            self.root_dir = root_dir
            self.transform = transform
            path = os.path.dirname(self.root_dir)
            self.classes = os.listdir(path)
            self.sample = sample
                               
        def __len__(self):
            lista = os.listdir(self.root_dir)
            return int(len(lista)/2)  # exige numero par de imagens

        def __getitem__(self, idx):

            lista = os.listdir(self.root_dir)
            root_list = [self.root_dir]*(len(lista))
            img_name = []
            for k in range(len(lista)):
                if lista[k] != 'Thumbs.db':
                    img_name.append(os.path.join(root_list[k], lista[k]))
        
            random.seed(seed1) # seed
            random.shuffle(img_name) 
            if self.sample == 'train':   
                img_name.reverse()
            image = io.imread(img_name[idx])
            
            if "uiuc" in self.root_dir or "umd" in self.root_dir:
                image = np.repeat(image[..., np.newaxis], 3, -1)    #criando mais dois canais de cores e copiando um nos demais (adaptando P&B para RGB)       
            
            img = Image.fromarray(image)
            label = basename(normpath(self.root_dir))
            label = self.classes.index(label)

            if self.transform:
                img = self.transform(img)

            return (img, label)
    
        def classes(self):

            return self.classes
        
        def paths(self, direc):
            
            lista = os.listdir(self.root_dir)
            root_list = [self.root_dir]*(len(lista))
            img_name = []
            for k in range(len(lista)):
                if lista[k] != 'Thumbs.db':
                    img_name.append(os.path.join(root_list[k], lista[k]))
        
            random.seed(seed1) # seed
            random.shuffle(img_name) 
            if self.sample == 'train':   
                img_name.reverse()
    
# -----------------------------------------------------------------------------------------------
# DATA AUGMENTATION AND NORMALIZATION
# -----------------------------------------------------------------------------------------------
    
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(200),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(200),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
# -----------------------------------------------------------------------------------------------
# LOCAL DE ARMAZENAMENTO DA BASE E AJUSTE DE PARÂMETROS
# -----------------------------------------------------------------------------------------------
  
    #root = r'D:\Datasets\Databases\Cysts4' #two classes cysts dataset (ks x r)
    #root = r'D:\Datasets\Databases\Cysts3' #two classes cysts dataset (k x s)
    #root = r'D:\Datasets\Databases\Cysts2' #three classes cysts dataset (k x s x r)
    #root = r'D:\Datasets\Databases\uiuc' #texture dataset (UIUC)
    #root = r'D:\Datasets\Databases\umd' #texture dataset (UMD)
    #root = r'D:\Datasets\Databases\Plants2' #texture dataset (Plants)
    #root = r'D:\Google Drive\UNICAMP\IC\KTH-TIPS\KTH-TIPS2-b' #base de texturas sem amostras separadas
    #root = r'D:\Datasets\databases\data\databases\fmd\image'
    root = r'D:\Datasets\databases\data\databases\dtd\images'

    model_name = 'resnet'
    #model_name = 'alexnet'
    model_type = 'ft'
    #model_type = 'conv'
    num_epochs = 15
    lr = 0.0001
    #alphas = [0.1,  0.2,  0.4, 0.6,  0.8,  1,  1.2,  1.4,  1.6,  1.8,  2]
    alphas = [1]
# -----------------------------------------------------------------------------------------------
# MANIPULANDO E CARREGANDO OS DATASETS
# -----------------------------------------------------------------------------------------------
    
    classes = os.listdir(root)
    data_dir = []
    image_datasets = {'train': [] , 'val': []}

    for i in range(len(classes)):
        data_dir.append(os.path.join(root,classes[i]))
    
    for i in range(len(classes)):
        image_datasets['val'].append(TextureDataset(data_dir[i], 'val', data_transforms['val']))
        image_datasets['train'].append(TextureDataset(data_dir[i], 'train', data_transforms['train']))

    image_datasets = {x: torch.utils.data.ConcatDataset(image_datasets[x]) for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------------------------
# FUNÇÃO PARA TREINAR A REDE
# -----------------------------------------------------------------------------------------------

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        epoch_loss = np.zeros(num_epochs)
        train_loss = np.zeros(num_epochs)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        cont = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    #print(inputs)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    train_loss[cont] = running_loss / dataset_sizes[phase]
                epoch_loss[cont] = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[cont], epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if phase == 'val': 
                    cont = cont + 1
                
            print()
        
        avg_train_loss.append(train_loss)
        avg_epoch_loss.append(epoch_loss)
        avg_best_acc.append(best_acc)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
            
# -----------------------------------------------------------------------------------------------
# AJUSTANDO PARÂMETROS DA REDE COM AJUSTE FINO
# -----------------------------------------------------------------------------------------------
    
    num_classes = len(class_names)
    
    if model_type == 'ft':
        if model_name == 'resnet':
    
            model_ft = models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        if model_name == 'alexnet':
        
            model_ft = models.alexnet(pretrained=True)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
# -----------------------------------------------------------------------------------------------
# TREINANDO A REDE E MOSTRANDO OS RESULTADOS
# -----------------------------------------------------------------------------------------------
        
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
        #visualize_model(model_ft)
        
# -----------------------------------------------------------------------------------------------
# AJUSTANDO PARÂMETROS DA REDE COM EXTRAÇÃO FIXA DE CARACTERÍSTICAS
# -----------------------------------------------------------------------------------------------

    if model_type == 'conv':    
        
        if model_name == 'resnet':
    
            model_conv = torchvision.models.resnet18(pretrained=True)
            for param in model_conv.parameters():
                param.requires_grad = False
            
            # Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = nn.Linear(num_ftrs, num_classes)
            
            model_conv = model_conv.to(device)

            criterion = nn.CrossEntropyLoss()

            # Observe that only parameters of final layer are being optimized as
            # opposed to before.
            optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=0.9)
        
        if model_name == 'alexnet':
        
            model_conv = models.alexnet(pretrained=True)
            for param in model_conv.parameters():
                param.requires_grad = False
            
            num_ftrs = model_conv.classifier[6].in_features
            model_conv.classifier[6] = nn.Linear(num_ftrs,num_classes)

            model_conv = model_conv.to(device)

            criterion = nn.CrossEntropyLoss()

            # Observe that only parameters of final layer are being optimized as
            # opposed to before.
            optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# -----------------------------------------------------------------------------------------------
# TREINANDO A REDE E MOSTRANDO OS RESULTADOS
# -----------------------------------------------------------------------------------------------
    
        model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)
        #visualize_model(model_conv)
    

# -----------------------------------------------------------------------------------------------
# MATRIZ DE CONFUSÃO
# -----------------------------------------------------------------------------------------------

    def conf_matrix(model_type):
        
        if model_type == 'conv':
            model = model_conv
        if model_type == 'ft':
            model = model_ft
        was_training = model.training
        model.eval()

        lbls = []
        prds = []

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                aux = labels.tolist()
                lbls = lbls + aux
            
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            
                aux = preds.tolist()
                prds = prds + aux

                model.train(mode=was_training)
            
        return lbls, prds

    lbls, prds = conf_matrix(model_type)

    mat = confusion_matrix(lbls, prds)
    avg_mat.append(mat)
    
    
# -----------------------------------------------------------------------------------------------
# FUNÇÃO QUE EXTRAI O VETOR DE CARACTERÍSTICAS DE CADA IMAGEM
# -----------------------------------------------------------------------------------------------

    class Img2Vec():

        def __init__(self, model_name='resnet', model_type='ft', cuda=False):
            self.device = torch.device("cuda" if cuda else "cpu")
            if model_type == 'ft':
                self.model = model_ft
            if model_type == 'conv':
                self.model = model_conv
            self.extraction_layer = self.model._modules.get('avgpool')
            self.model = self.model.to(self.device)

            self.model.eval()

        def get_vec(self, img):
            if model_name == 'resnet':
                my_embedding = torch.zeros(1, 512, 1, 1)
            if model_name == 'alexnet':
                my_embedding = torch.zeros(1, 256, 6, 6)
        
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(img)
            h.remove()

            return my_embedding
        
# -----------------------------------------------------------------------------------------------
# EXTRAINDO MATRIZ DE FEATURES E DE RÓTULOS
# -----------------------------------------------------------------------------------------------
    
    img2vec = Img2Vec(model_name, model_type)
    
    if model_name == 'resnet':
        vec_length = 512
    if model_name == 'alexnet':
        vec_length = 256

    samples = image_datasets['train'].__len__()  # Amount of samples to take from input path
    samples2 = image_datasets['val'].__len__()

    # Matrix to hold the image vectors
    vec_mat = np.zeros((samples, vec_length))
    labels = np.zeros(samples)
    vec_mat2 = np.zeros((samples2, vec_length))
    labels2 = np.zeros(samples2)

    for i in range(samples):
    
        img, label = image_datasets['train'].__getitem__(i)
        img = img.unsqueeze(0)
        vec = img2vec.get_vec(img)
        labels[i] = label
        vec_mat[i, :] = vec[0,:,0,0]
    
    for i in range(samples2):
    
        img2, label2 = image_datasets['val'].__getitem__(i)
        img2 = img2.unsqueeze(0)
        vec2 = img2vec.get_vec(img2)
        labels2[i] = label2
        vec_mat2[i, :] = vec2[0,:,0,0]
        
        

    #pd.DataFrame(vec_mat).to_csv(r'D:\HBoost_results\vec_mat.csv')
    #pd.DataFrame(vec_mat2).to_csv(r'D:\HBoost_results\vec_mat2.csv')
    #pd.DataFrame(labels).to_csv(r'D:\HBoost_results\labels.csv')
    #pd.DataFrame(labels2).to_csv(r'D:\HBoost_results\labels2.csv')
        
# -----------------------------------------------------------------------------------------------
# APLICANDO PCA
# -----------------------------------------------------------------------------------------------
        
    #pca = PCA(48)

    #vec_mat_ = vec_mat
    #vec_mat2_ = vec_mat2
    
    #vec_mat = pca.fit_transform(vec_mat) #não aplicar por enquanto
    #vec_mat2 = pca.fit_transform(vec_mat2)
        
# -----------------------------------------------------------------------------------------------
# APLICANDO CLASSIFICADOR SVM, KNN, LDA E RF
# -----------------------------------------------------------------------------------------------

    clf = LinearDiscriminantAnalysis().fit(vec_mat, labels)
    avg_clf1.append(clf.score(vec_mat2, labels2))
    print('Best val Acc (LDA): {:4f}'.format(clf.score(vec_mat2, labels2)))
    scores2 = clf.predict_proba(vec_mat2)
    scores_train2 = clf.predict_proba(vec_mat)
    clf1 = clf.predict(vec_mat2)

    clf = svm.SVC(kernel='linear', C=1, probability=True).fit(vec_mat, labels)
    avg_clf2.append(clf.score(vec_mat2, labels2))
    print('Best val Acc (SVM): {:4f}'.format(clf.score(vec_mat2, labels2)))
    scores = clf.predict_proba(vec_mat2)
    scores_train = clf.predict_proba(vec_mat)
    clf2 = clf.predict(vec_mat2)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(vec_mat, labels)
    avg_clf3.append(clf.score(vec_mat2, labels2))
    print('Best val Acc (RF): {:4f}'.format(clf.score(vec_mat2, labels2)))
    clf3 = clf.predict(vec_mat2)

    clf = NearestCentroid().fit(vec_mat, labels)
    avg_clf4.append(clf.score(vec_mat2, labels2))
    print('Best val Acc (KNN): {:4f}'.format(clf.score(vec_mat2, labels2)))
    clf4 = clf.predict(vec_mat2)

# -----------------------------------------------------------------------------------------------
# LBP
# -----------------------------------------------------------------------------------------------   
    
    class LocalBinaryPatterns:
        def __init__(self, numPoints, radius):
		# store the number of points and radius
            self.numPoints = numPoints
            self.radius = radius
 
        def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
            lbp = feature.local_binary_pattern(image, self.numPoints,
			    self.radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
    			bins=np.arange(0, self.numPoints + 3),
			    range=(0, self.numPoints + 2))
 
		# normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
            return hist 
    
# -----------------------------------------------------------------------------------------------
# CALCULANDO DISTANCIAS ENTRE IMAGEM E CLASSES
# -----------------------------------------------------------------------------------------------  
    
    def Nmaxelements(list1, N=5): 
        final_list = [] 
  
        for i in range(0, N):  
            max1 = 0
          
            for j in range(len(list1)):      
                if list1[j] > max1: 
                    max1 = list1[j]; 
                  
            list1.remove(max1); 
            final_list.append(max1) 
          
        return final_list 

    since = time.time()
    pesos = []
    imagePaths = []
    roots = []
    roots1 = os.listdir(root)
    for k in range(len(roots1)):
        roots.append(os.path.join(root, roots1[k]))

    desc = LocalBinaryPatterns(24, 8)
    dists = []
    tamanhos = []
    
    for root_dir in roots:
        lista = os.listdir(root_dir)
        root_list = [root_dir]*(len(lista))
        tamanhos.append(len(lista)//2) #número de imagens por classe
        img_name = []
        for k in range(len(lista)):
            if lista[k] != 'Thumbs.db':
                img_name.append(os.path.join(root_list[k], lista[k]))
        
        random.seed(seed1) # seed
        random.shuffle(img_name)    
        img_name.reverse()
        imagePaths = imagePaths + img_name[:len(img_name)//2]
        
    tam_totais = np.cumsum(tamanhos)
    tam_totais = np.insert(tam_totais, 0, 0)
    
    histogramas = []

    for imagePath in imagePaths:

    # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        histogramas.append(hist)
        
    dists = []

    for histograma in histogramas: #fixa uma imagem de referencia
        distances = []
        
        for i in range(len(tamanhos)): #rodando as classes
            dist = []
            
            for j in range(tamanhos[i]): #rodando as imagens de cada classe
                dist.append(np.linalg.norm(histogramas[j + tam_totais[i]] - histograma))

            dista = Nmaxelements(dist, 5)
            dist = np.average(dista,axis=0)
            distances.append(dist)
            
        dists.append(distances)
    
# -----------------------------------------------------------------------------------------------
# CALCULANDO PESOS E ACURÁCIA
# -----------------------------------------------------------------------------------------------  
        
        peso = np.zeros(len(distances))
        for k in range(len(distances)):
            aux = distances.copy()
            del aux[k]
            for j in range(len(aux)):
                peso[k] = peso[k] + aux[j]
            peso[k] = peso[k]/distances[k]
    
        pesos.append(peso)
    
    result = []
    for i in range(len(pesos)):
        result.append(pesos[i] * scores[i])
    
    predict = []
    for k in range(len(result)):
        result[k] = result[k].tolist()
        predict.append(result[k].index(max(result[k])))

    match = 0
    for j in range(len(predict)):
        if predict[j] == labels[j]:
            match = match + 1
    match = match/len(predict)

    avg_lbp.append(match)
    print('Best val Acc (LBP): {:4f}'.format(match))
    
# -----------------------------------------------------------------------------------------------
# LBP COM NN (SVM)
# -----------------------------------------------------------------------------------------------
    
    N, D_in, H, D_out = len(lbls), len(roots)*2, len(roots)*3*2, len(roots)

    in_feature = []
    scores_train = np.asarray(scores_train, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    in_feature = np.hstack((scores_train, dists))
    x = torch.FloatTensor(in_feature) 
    
    test_feature = []
    scores = np.asarray(scores, dtype=np.float32)
    test_feature = np.hstack((scores, dists))
    x_test = torch.FloatTensor(test_feature) 

    y = []
    for i in range(len(labels)):
        aux = [0]*len(roots)
        aux[int(labels[i])] = 1
        y.append(aux)
            
    y = torch.FloatTensor(y)
    
    model = torch.nn.Sequential(torch.nn.Linear(D_in, H),torch.nn.ReLU(),torch.nn.Linear(H, D_out), nn.Softmax(dim=1))

    torch.set_grad_enabled(True) 

    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.CrossEntropyLoss()

    #learning_rate = 0.1 #funciona com cistos
    learning_rate = 1.1 #funciona com plantas
    #learning_rate = 1.6 #funciona com uiuc e umd
    
    #num_epochs2 = num_epochs*20 #funciona com cistos
    num_epochs2 = num_epochs*150 #funciona com plantas
    #num_epochs2 = num_epochs*800 #funciona com uiuc e umd
    
    matches = []
    predicts = []
    for t in range(num_epochs2):

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
#        if t % 10 == 9:
        prdct=[]
        y_pred = y_pred.tolist()
        for k in range(len(y_pred)):
            prdct.append(y_pred[k].index(max(y_pred[k])))
            
        match = 0
        for j in range(len(prdct)):
            if prdct[j] == labels[j]:
                match = match + 1
        match = match/len(prdct)
                    
            #print('Epoch {}/{}'.format(t,num_epochs2))
            #print('------------')
            #print('train Loss: {:.4f} Acc: {:.4f}'.format(loss.item(), match))
            
        y_test = model(x_test)
        loss_test = loss_fn(y_test, y)
            
        prdct_test=[]
        y_test = y_test.tolist()
        for k in range(len(y_test)):
            prdct_test.append(y_test[k].index(max(y_test[k])))
        predicts.append(prdct_test)
            
        match = 0
        for j in range(len(prdct_test)):
            if prdct_test[j] == labels[j]:
                match = match + 1
        match = match/len(prdct_test)
        matches.append(match)
            #print('val Loss: {:.4f} Acc: {:.4f}'.format(loss_test.item(), match))
            #print(' ')

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    
    best_acc = max(matches)
    best_predict = predicts[matches.index(max(matches))]
    avg_lbpnn.append(best_acc)
    print('Best val Acc (LBPNN): {:4f}'.format(best_acc))
    
    
# -----------------------------------------------------------------------------------------------
# LBP COM NN (SVM + LDA)
# -----------------------------------------------------------------------------------------------
  
    N, D_in, H, D_out = len(lbls), len(roots)*3, len(roots)*3*2, len(roots)

    in_feature = []
    scores_train2 = np.asarray(scores_train2, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    in_feature = np.hstack((scores_train, scores_train2, dists))
    x = torch.FloatTensor(in_feature) 
    
    test_feature = []
    scores2 = np.asarray(scores2, dtype=np.float32)
    test_feature = np.hstack((scores, scores2, dists))
    x_test = torch.FloatTensor(test_feature) 

    y = []
    for i in range(len(labels)):
        aux = [0]*len(roots)
        aux[int(labels[i])] = 1
        y.append(aux)
            
    y = torch.FloatTensor(y)

    model = torch.nn.Sequential(torch.nn.Linear(D_in, H),torch.nn.ReLU(),torch.nn.Linear(H, D_out), nn.Softmax(dim=1))

    loss_fn = torch.nn.MSELoss()

    matches = []
    predicts = []
    for t in range(num_epochs2):

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
#        if t % 10 == 9:
        prdct=[]
        y_pred = y_pred.tolist()
        for k in range(len(y_pred)):
            prdct.append(y_pred[k].index(max(y_pred[k])))
        match = 0
        for j in range(len(prdct)):
            if prdct[j] == labels[j]:
                match = match + 1
        match = match/len(prdct)
                    
            #print('Epoch {}/{}'.format(t,num_epochs2))
            #print('------------')
            #print('train Loss: {:.4f} Acc: {:.4f}'.format(loss.item(), match))
            
        y_test = model(x_test)
        loss_test = loss_fn(y_test, y)
            
        prdct_test=[]
        y_test = y_test.tolist()
        for k in range(len(y_test)):
            prdct_test.append(y_test[k].index(max(y_test[k])))
        predicts.append(prdct_test)
            
        match = 0
        for j in range(len(prdct_test)):
            if prdct_test[j] == labels[j]:
                match = match + 1
        match = match/len(prdct_test)
        matches.append(match)
            #print('val Loss: {:.4f} Acc: {:.4f}'.format(loss_test.item(), match))
            #print(' ')

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    
    best_acc2 = max(matches)
    best_predict2 = predicts[matches.index(max(matches))]
    avg_lbpnn2.append(best_acc2)
    print('Best val Acc (LBPNN2): {:4f}'.format(best_acc2))
    
# -----------------------------------------------------------------------------------------------
# ENSEMBLES
# -----------------------------------------------------------------------------------------------
    
    ens_list2 = []
    ens_list3 = []
    result = []
    ensemble = []
    ensemb = []
    
    ens_list = [y_fv, y_fcfv, prds, clf1, clf2, clf3, clf4, predict, best_predict]
    
    for i in range(len(ens_list)):    
        ensemb.append(np.mean(np.array(ens_list[i]) == np.array(labels2)))
    minIndex = ensemb.index(min(ensemb))
    
    del ens_list[minIndex] #removendo pior resultado do ensemble
    del ensemb[minIndex]
    minIndex = ensemb.index(min(ensemb))
    del ens_list[minIndex] #removendo segundo pior resultado do ensemble
    
    def most_frequent(List): 
        return max(set(List), key = List.count) 
    
    for i in range(2,len(ens_list)+1):
        ens_list2.append(list(itertools.combinations(ens_list, i))) #combinações de ensembles 2 a 2, 3 a 3, ...
   
    for i in range(len(ens_list2)):
        ens_list3 = ens_list3 + ens_list2[i] #concatenando combinações
        
    for i in range(len(ens_list3)):    
        for a in range(len(prds)):
            ens_list4 = [item[a] for item in ens_list3[i]]
            result.append(most_frequent(ens_list4))
        ensemble.append(np.mean(np.array(result) == np.array(labels2)))
        result = []
        

    print('Best val Acc (ENSEMBLE): {:4f}'.format(max(ensemble)))
    avg_ens.append(max(ensemble))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    avg_time.append(time_elapsed)

    
# -----------------------------------------------------------------------------------------------
# APLICANDO BAGGINGS PARA MONTAR MATRIZ DE PODA
# -----------------------------------------------------------------------------------------------
  

    def shape(ndarray: Union[List, float]) -> Tuple[int, ...]:
        if isinstance(ndarray, list):
            # More dimensions, so make a recursive call
            outermost_size = len(ndarray)
            row_shape = shape(ndarray[0])
            return (outermost_size, *row_shape)
        else:
            # No more dimensions, so we're done
            return ()

    classifiers = []
    comb_feat = []

    classifiers.append(SVC(kernel='linear', C=1, probability=True))
    #classifiers.append(DecisionTreeClassifier(criterion="entropy",max_depth=1))
    #classifiers.append(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0))
    classifiers.append(MLPClassifier(solver='sgd', hidden_layer_sizes=(6,), random_state=1))

    np_feat = np.array([np.array(aux) for aux in features])
    
    for i in range(vec_mat.shape[0]):
        comb_feat.append(np.concatenate((vec_mat[i],np_feat[i]), axis=None))
    comb_np = np.array([np.array(aux) for aux in comb_feat])
    
    X_train, X_test, y_train, y_test = train_test_split(np_feat, labels, test_size=0.5, random_state=42) #50% test, 50% train
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42) #splitting test again: 30% test, 20% validation

    #X_train, X_test, y_train, y_test = vec_mat, vec_mat2, labels, labels2
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42) #splitting again: 40% test, 10% validation
    
    lines = 0
    matrix = [] #armazena matriz de poda
    for est in classifiers:
        clf_fv = BaggingClassifier(estimator=est,n_estimators=10, random_state=22).fit(X_train, y_train)
        matrix.append(clf_fv) #baggings com features de multilevel pooling
        #matrix.append([est.fit(X_train, y_train)])
        lines += 1
        

    for i in range(len(matrix)):
        matrix[i] = list(filter(None, matrix[i])) #remove None values

# -----------------------------------------------------------------------------------------------
# HBOOST - CALCULANDO ENTROPIAS
# -----------------------------------------------------------------------------------------------

    def sigma(classifiers, X, y, i):
        cont = 0
        for classifier in classifiers:
            prediction = classifier.predict(X)
            if prediction[i] == y[i]:
                cont += 1
        return cont
    
    def sigma_score(classifiers, X, y, i):
        cont = 0
        for classifier in classifiers:
            prediction = classifier.predict(X)
            if prediction[i] == y[i]:
                cont += classifier.score(X, y)
        return cont

    def entropia(classifiers, X_val, y_val):
        E = 0
        for i in range(len(y_val)):
            L = sigma(classifiers, X_val, y_val, i)
            E += min(L, len(classifiers) - L) / len(classifiers)
        return 2*E/len(y_val)

    def shannon(classifiers, X_val, y_val):
        E = 0
        for i in range(len(y_val)):
            L = sigma(classifiers, X_val, y_val, i)
            P = 2*min(L, len(classifiers) - L) / len(y_val)
            if P != 0:
                E += P * np.log2(P)
        return -E

    def renyi(classifiers, X_val, y_val, alpha):
        alfa = alpha # FATOR ALFA (alfa=1 ENTROPIA DE SHANNON)
        H = 0
        for i in range(len(y_val)):
            L = sigma(classifiers, X_val, y_val, i) # usando número de classificadores que acertam
            #L = sigma_score(classifiers, X_val, y_val, i) # usando soma das scores
            P = 2*min(L, len(classifiers) - L) / len(y_val)
            H += P**alfa
        
        if alfa == 1:
            H = shannon(classifiers, X_val, y_val)
        else:
            if H != 0:
                H = 1 / (1 - alfa) * np.log2(H)
        return H
      
# -----------------------------------------------------------------------------------------------
# HBOOST - PRUNING MATRIZ DE CLASSIFICADORES
# -----------------------------------------------------------------------------------------------
    avg_aux = []
    for alpha in alphas:
        pruned = []
        for linha in range(len(matrix)): 
            entropias = []
            classificadores = []
            if len(matrix[linha]) == 1: #um único tipo de classificador
                classificadores.append(matrix[linha])
                #entropias.append(entropia(matrix[linha], X_val, y_val)) #ENTROPIA ORIGINAL
                entropias.append(renyi(matrix[linha], X_val, y_val, alpha)) #ENTROPIA DE RENYI/SHANNON
            else:
                for i in range(2, len(matrix[linha]) + 1): #pelo menos 2 tipos de classificadores
                    for subset in itertools.combinations(matrix[linha], i):
                        classificadores.append(subset)
                        #entropias.append(entropia(subset, X_val, y_val)) #ENTROPIA ORIGINAL
                        entropias.append(renyi(subset, X_val, y_val, alpha)) #ENTROPIA DE RENYI/SHANNON


            max_E = entropias.index(max(entropias)) #posição da entropia máxima
            pruned.append(classificadores[max_E]) # combinação de classificadores que gera entropia máxima
            print('-----------------------------------')
            print(pruned)
    
# -----------------------------------------------------------------------------------------------
# HBOOST - VOTAÇÃO E CLASSIFICAÇÃO FINAL
# -----------------------------------------------------------------------------------------------
        
        names = list(map(str, range(len(pruned) + 1)))
        voting = [] 
        for linha in pruned:
            vote_name = list(zip(names, linha))
            voting.append(VotingClassifier(estimators=vote_name))
    
        predictions = []
        for linha in voting:
            linha.fit(X_train, y_train)
            predictions.append(linha.predict(X_test))
            print(linha.score(X_test, y_test))
        print(predictions)
        print(y_test)

        y_hboost = []
        for i in range(len(y_test)): #rodando imagens
            C = []
            for classe in range(len(classes)): #rodando classes
                aux = 0
                for j in range(len(pruned)): #rodando linhas da matriz
                    if predictions[j][i] == classe:
                        #aux += math.log(1/pruned_weights[j])
                        aux += 1
                C.append(aux)
            max_C = C.index(max(C)) #classificação final
            y_hboost.append(max_C)

        acertos = sum(x == y for x, y in zip(y_hboost, y_test.tolist()))
        print('Best val Acc (Ours): {:.2%}'.format(acertos/len(y_hboost))) #acurácia final
        avg_aux.append(acertos/len(y_hboost))
    #avg_alphas.append(avg_aux.copy()) #armazenando as acurácias para cada alpha
    
    



