import random

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision.models.resnet import resnet50
from torchinfo import summary

from easyfsl.datasets import FewShotDataset
from easyfsl.methods import FewShotClassifier


class MyDataset(FewShotDataset):
    """
    Crea gli split del dataset. Per eseguire la cross validation, uno di questi dovrà essere usato come validation,
    gli altri devono essere uniti per ottenere il train usando la classe Concat.
    """
    def __init__(self, path: Path, n_folds: int, split: int, mode: str, transforms: Compose, seed: int):
        """
        Crea il dataset a partire dal percorso indicato.

        :param path:        percorso della cartella contenente il dataset.
        :param n_folds:     numero di fold per crossvalidation.
        :param split:       indice per identificare quale split stiamo usando.
        :param mode:        per dire se stiamo facendo 'train' o 'eval'.
        :param transforms:  le trasformazioni da applicare all'immagine.
        :param seed:        per riproducibilità.
    """
        self.path = path
        self.img_dir = path / 'images'
        self.annos = pd.read_csv(path / 'subset_10_images.csv', index_col='image_id')  # leggo il csv con pandas
        self.n_folds = n_folds
        self.split = split
        self.mode = mode
        self.seed = seed
        # in eval mode vogliamo usare il test set, che è stato ottenuto dalla coda del dataset totale (maggiori info
        # sotto _get_ids_in_split())
        if self.mode == 'eval':
            self.split = self.n_folds
        self.transforms = transforms
        self.all_ids = self.annos.index.tolist()    # lista di tutti gli image_id
        #
        #
        # QUI CAMBIA RISPETTO A PRIMA PERCHE' LE CLASSI DIVENTANO I PRODUCT_ID UNIVOCI PER OGNI PRODOTTO
        #
        #
        self.classes = sorted(self.annos['item_id'].unique().tolist())   # tutte le classi del dataset
        self.mapping = {item_id: i for i, item_id in enumerate(self.classes)}
        # ciclicamente solo una porzione del dataset completo viene utilizzata (sarà usata come validation set)
        self.image_ids = self._get_ids_in_split()
        self.labels = self.annos.loc[self.image_ids, 'item_id'].to_list()  # lista con le classi dello split
        self.labels = [self.mapping[i] for i in self.labels]

    def __len__(self):
        """Numero di esempi nel dataset"""
        return len(self.labels)

    def __getitem__(self, index):
        """
        Restituisce la tuple (image, label)

        :param index:   indice dell'elemento da restituire
        :return:
        """
        image_id = self.image_ids[index]
        # recupero il filename dell'immagine nella colonna 'path' e la apro
        image_path = self.img_dir / self.annos.loc[image_id, 'path']
        image = cv2.imread(str(image_path))

        if self.transforms:
            image = self.transforms(image)  # applico le trasformazioni desiderate

        label = self.labels[index]
        return image, label

    def _get_ids_in_split(self):
        """
        Restituisce la lista dei percorsi delle immagini nello split attuale. In particolare, la funzione suddivide
        il dataset completo in n_folds+1 parti (bilanciate) così da poter fare crossvalidation usando stratified
        k-folding e al tempo stesso tenere da parte un hold out set per fare test.

        QUI SIAMO NEL CASO FSL, QUINDI VOGLIAMO IMBASTIRE UN EPISODIC LEARNING, PER CUI LO SPLIT AVVERRA' IN MODO
        CHE LE CLASSI DI TRAIN E QUELLE DI TEST SIANO DISGIUNTE.
        """
        # a senso non dovrebbe cambiare niente rispetto al caso semplice, ovviamente a questo giro all_classes sono
        # gli item_id invece dei product_type
        classes_per_split = len(self.classes) // (self.n_folds + 1)
        start = self.split * classes_per_split
        if self.mode == 'train':
            end = start + classes_per_split
        elif self.mode == 'eval':
            end = None
        random.seed(self.seed)
        classes_in_split = random.shuffle(self.classes)[start: end]
        ids_in_split = self.annos.index[self.annos['item_id'].isin(classes_in_split)].tolist()

        return ids_in_split

    def set_transforms(self, transforms):
        """
        Consente di modificare le trasformazioni da apportare. Serve perchè questa classe crea gli split che a
        rotazione saranno i validation set, quindi applica le val_transforms. Il train set è restituito invece da
        Concat successivamente, quindi le trasformazioni vanno cambiate.
        """

        self.transforms = transforms

    def get_labels(self):
        return self.labels

    def number_of_classes(self):
        return len(self.classes)


class Concat(Dataset):
    """
    La classe MyDataset prende l'intero dataset e crea n_folds+1 split bilanciati. I primi n_folds verranno usati
    per fare stratified k-fold crossvalidation, l'ultimo viene tenuto come test. Per fare k-fold, ciclicamente tengo
    uno degli split come validation e gli altri come train. Questa classe consente di unire due split in un unico
    dataset e verrà utilizzata per ottenre i dataset di train.
    """

    def __init__(self, datasets: list):
        """
        Unisce i dataset che gli vengono passati come lista.

        :param datasets:    lista dei dataset da unire.
        """
        self.datasets = datasets
        self.lenghts = [len(i) for i in self.datasets]
        # array contenente gli estremi destri di ogni dataset
        self.borders = np.cumsum(self.lenghts)

    def __len__(self):

        return sum(self.lenghts)

    def __getitem__(self, index):
        for i, b in enumerate(self.borders):
            if index < b:
                if i > 0:
                    index -= self.borders[i - 1]
                # return interrompe il ciclo for, quindi non c'è bisogno di controllare di essere tra i due
                # estremi del dataset, basta ritornare il primo in cui si 'entra', traslando gli indici se i>0
                return self.datasets[i][index]


def train_one_epoch(model: FewShotClassifier, dataloader: DataLoader, epoch: int, optimizer: torch.optim.Optimizer,
                    criterion, writer: SummaryWriter, device):
    """
    Allena la rete per un'epoca.

    :param model:       il modello(già sul giusto device).
    :param dataloader:  il dataloader del dataset di train.
    :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
    :param optimizer:   per aggiornare i parametri.
    :param criterion:   per la loss.
    :param writer:      per salvare le metriche.
    :return:
    """
    model.train()      # modalità train
    optimizer.zero_grad()   # svuoto i gradienti

    n_episodes = len(dataloader)
    progress = tqdm(dataloader, total=n_episodes, leave=False, desc='COMPLETED EPISODES')
    epoch_loss = 0.0    # inizializzo la loss
    epoch_correct = 0   # segno le prediction corrette della rete per poi calcolare l'accuracy
    tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
    for sample in progress:
        support_images, support_labels, query_images, query_labels = sample
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)

        batch_cases = query_images.shape[0]  # numero di sample nel batch
        tot_cases += batch_cases  # accumulo il numero totale di sample

        # output della rete
        model.process_support_set(support_images, support_labels)   # usa il support set per fine tuning
        classification_scores = model(query_images)
        logits = model(images)  # output della rete prima di applicare softmax
        outputs = F.softmax(logits, dim=1)  # class probabilities
        # il risultato di softmax viene interpretato con politica winner takes all
        batch_decisions = torch.argmax(outputs, dim=1)

        # loss del batch e backward step
        batch_loss = criterion(classification_scores, query_labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # accumulo le metriche di interesse
        epoch_loss += batch_loss.item()    # avendo usato reduction='sum' nella loss qui sto sommando la loss totale
        batch_correct = torch.sum(batch_decisions == labels)    # risposte corrette per il batch attuale
        epoch_correct += batch_correct.item()      # totale risposte corrette sull'epoca

        postfix = {'batch_mean_loss': batch_loss.item()/batch_cases,
                   'batch_accuracy': (batch_correct.item()/batch_cases) * 100.0}
        progress.set_postfix(postfix)

    epoch_mean_loss = epoch_loss / tot_cases        # loss media sull'epoca
    epoch_accuracy = (epoch_correct / tot_cases) * 100.0        # accuracy sull'epoca (%)
    writer.add_scalar(f'Loss/Train', epoch_mean_loss, epoch + 1)
    writer.add_scalar(f'Accuracy/Train', epoch_accuracy, epoch + 1)
