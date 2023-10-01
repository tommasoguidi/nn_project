import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAffine, ToTensor, Normalize
from torchvision.models.resnet import resnet50, resnet18
from torchinfo import summary


class MyDataset(Dataset):
    """
    Crea gli split del dataset. Per eseguire la cross validation, uno di questi dovrà essere usato come validation,
    gli altri devono essere uniti per ottenere il train usando la classe Concat.
    """
    def __init__(self, path: Path, n_folds: int, split: int, mode: str, transforms: Compose, method: str, seed: int):
        """
        Crea il dataset a partire dal percorso indicato.

        :param path:        percorso della cartella contenente il dataset.
        :param n_folds:     numero di fold per crossvalidation.
        :param split:       indice per identificare quale split stiamo usando.
        :param mode:        per dire se stiamo facendo 'train' o 'eval'.
        :param transforms:  le trasformazioni da applicare all'immagine.
        :param method:      naive o moe.
        :param seed:        per riproducibilità.
        """
        self.path = path
        self.img_dir = path / 'images'
        self.annos = pd.read_csv(path / 'subset_10_images.csv', index_col='image_id')  # leggo il csv con pandas
        self.n_folds = n_folds
        self.split = split
        self.mode = mode
        self.method = method
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
        self.all_classes = sorted(self.annos['item_id'].unique().tolist())   # tutte le classi del dataset
        self.all_super_classes = sorted(self.annos['product_type'].unique().tolist())     # classi del caso semplice
        self.mapping = {}
        if self.method == 'naive':
            for i, item_id in enumerate(self.all_classes):
                self.mapping[item_id] = i
            self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        else:
            for i, super_class in enumerate(self.all_super_classes):
                # tutte le righe del dataframe relative ad una categoria merceologica
                _listings_in_superclass = self.annos[self.annos['product_type'] == super_class]
                # lista degli item_id univoci della superclass (metterli in ordine ovviamente non cambia niente)
                _items_in_superclass = sorted(_listings_in_superclass['item_id'].unique().tolist())
                # identifier sarà l'int che identifica la superclass i
                self.mapping[super_class] = {'identifier': i}
                # sotto la superclass, associo un intero a ciascuno degli item univoci
                for j, item_id in enumerate(_items_in_superclass):
                    self.mapping[super_class][item_id] = j
        # ciclicamente solo una porzione del dataset completo viene utilizzata (sarà usata come validation set)
        self.image_ids = self._get_ids_in_split()

    def __len__(self):
        """Numero di esempi nel dataset"""
        return len(self.image_ids)

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

        # a questo punto del progetto l'interesse è quello di allenare un classificatore in grado di riconoscere
        # l'oggetto in particolare. la item_label quindi sarà il valore 'item_id', mentre la super_class_label è
        # 'product_type', che vengono mappati al loro valore intero di riferimento e trasformati in un tensore
        super_class_label = self.annos.loc[image_id, 'product_type']
        item_label = self.annos.loc[image_id, 'item_id']
        if self.method == 'naive':
            item_label = torch.tensor(self.mapping[item_label], dtype=torch.long)
            return image, item_label, str(image_path)
        else:
            item_label = torch.tensor(self.mapping[super_class_label][item_label], dtype=torch.long)
            super_class_label = torch.tensor(self.mapping[super_class_label]['identifier'], dtype=torch.long)
            return image, super_class_label, item_label, str(image_path)

    def _get_ids_in_split(self):
        """
        Restituisce la lista dei percorsi delle immagini nello split attuale. In particolare, la funzione suddivide
        il dataset completo in n_folds+1 parti (bilanciate) così da poter fare crossvalidation usando stratified
        k-folding e al tempo stesso tenere da parte un hold out set per fare test
        """
        # a senso non dovrebbe cambiare niente rispetto al caso semplice, ovviamente a questo giro all_classes sono
        # gli item_id invece dei product_type
        ids_in_split = []
        for _class in self.all_classes:
            # metto in una lista tutte gli image_id di una determinata classe
            ids_in_class = self.annos.index[self.annos['item_id'] == _class].tolist()
            random.seed(self.seed)
            random.shuffle(ids_in_class)
            # splitto in n_folds + 1 per avere un hold out su cui fare il test (l'ultimo degli split)
            ids_per_split = len(ids_in_class) // (self.n_folds + 1)     # n di immagini che metto nello split
            # se siamo in train mode self.split è scandito dal loop del kfold 0,1,...,k
            # se siamo in eval mode allora in __init__ self.split è stato settato pari a n_folds, il che mettendo
            # end=None equivale a prendere la coda di ids_in_class
            start = self.split * ids_per_split
            if self.mode == 'train':
                end = start + ids_per_split
            elif self.mode == 'eval':
                end = None

            ids_in_split.extend(ids_in_class[start: end])   # aggiungo e ritorno gli id selezionati

        return ids_in_split

    def set_transforms(self, transforms):
        """
        Consente di modificare le trasformazioni da apportare. Serve perchè questa classe crea gli split che a
        rotazione saranno i validation set, quindi applica le val_transforms. Il train set è restituito invece da
        Concat successivamente, quindi le trasformazioni vanno cambiate.
        """

        self.transforms = transforms


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


class Head(nn.Module):
    """Questa è la classe base delle classification heads usate nel caso 'moe'."""
    def __init__(self, in_features: int, classes: int,  depth: int):
        super().__init__()
        self.depth = depth
        if self.depth == 3:
            self.layer1 = nn.Linear(in_features, 4096)
            self.layer2 = nn.Linear(4096, 2048)
            self.layer3 = nn.Linear(2048, classes)
        else:
            self.layer1 = nn.Linear(in_features, classes)

    def forward(self, x):
        if self.depth == 3:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            # x = F.dropout(x)
            logits = self.layer3(x)  # output della rete prima di applicare softmax
        else:
            logits = self.layer1(x)  # output della rete prima di applicare softmax

        return logits


class MyResNet(nn.Module):
    """Riscrivo il forward della ResNet originaria per prelevare il vettore delle features"""
    def __init__(self, backbone, num_super_classes, pretrained, weights, device):
        super().__init__()
        self.device = device
        if backbone == 'resnet':
            if pretrained:
                self.resnet = resnet50(num_classes=num_super_classes)
                model_state = torch.load(weights, map_location=self.device)
                self.resnet.load_state_dict(model_state["model"])
                # congelo i parametri per allenare solo il layer finale
                for p in self.resnet.parameters():
                    p.requires_grad = False
            else:
                self.resnet = resnet50(weights='DEFAULT', progress=True)    # carico i pesi di imagenet
                # congelo i parametri tranne quelli dell'ultimo bottleneck
                blocks = list(self.resnet.children())
                for b in blocks[:-3]:
                    for p in b.parameters():
                        p.requires_grad = False
                # modifico il linear layer per la classificazione della super classe
                self.resnet.fc = nn.Linear(2048, num_super_classes)
        elif backbone == 'resnet18':
            if pretrained:
                self.resnet = resnet18(num_classes=num_super_classes)
                model_state = torch.load(weights, map_location=self.device)
                self.resnet.load_state_dict(model_state["model"])
                # congelo i parametri per allenare solo il layer finale
                for p in self.resnet.parameters():
                    p.requires_grad = False
            else:
                self.resnet = resnet18(weights='DEFAULT', progress=True)    # carico i pesi di imagenet
                # congelo i parametri tranne quelli dell'ultimo bottleneck
                # blocks = list(self.resnet.children())
                # for b in blocks[:-3]:
                #     for p in b.parameters():
                #         p.requires_grad = False
                # modifico il linear layer per la classificazione della super classe
                self.resnet.fc = nn.Linear(512, num_super_classes)

    def forward(self, x):
        # implementazione ufficiale pytorch del forward, modificata per ritornare il feature vector
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        feature_vector = torch.flatten(x, 1)
        x = self.resnet.fc(feature_vector)

        return x, feature_vector


class MoE(nn.Module):
    """
    Il modulo è composto dalla resnet preallenata su imagenet e una lista di classification heads, una per ciascuna
    classe.
    """
    def __init__(self, backbone: str, num_super_classes: int, len_item_classes: list,
                 pretrained: bool, weights: Path, device: str, depth: int):
        """

        :param num_super_classes:   'resnet' o 'resnet18'.
        :param num_super_classes:   numero delle super classi.
        :param len_item_classes:    lista con il numero di prodotti per ciascuna super classe
                                    -> len_item_classes[i] è il numero di prodotti della classe i.
        :param pretrained:          se caricare il modello preallentato di resnet.
        :param weights:             percorso dei pesi da caricare.
        :param device:              cpu o gpu.
        :param depth:               strati della head.
        """
        super().__init__()
        self.num_super_classes = num_super_classes
        self.len_item_classes = len_item_classes
        # metto la mia resnet con forward modificato
        self.resnet = MyResNet(backbone, self.num_super_classes, pretrained, weights, device)
        # creo un'istanza della classe Head() per ogni super classe e la aggiungo alla ModuleList
        # la i-esima istanza ha come in_features la dimensione delle feature uscenti dalla resnet dopo flatten() (2048)
        # e come classes il numero dei prodotti appartenenti alla i-esima super class
        if backbone == 'resnet':
            self.heads = nn.ModuleList([Head(2048, self.len_item_classes[i], depth) for i in range(self.num_super_classes)])
        elif backbone == 'resnet18':
            self.heads = nn.ModuleList([Head(512, self.len_item_classes[i], depth) for i in range(self.num_super_classes)])

    def forward(self, x, super_class):
        # il metodo forward() di resnet è stato modificato per ritornare anche il feature vector
        # il forward di moe avviene su un singolo evento e non su un batch
        super_class_logits, feature_vector = self.resnet.forward(x)
        super_class_output = F.softmax(super_class_logits, dim=1)  # class probability
        super_class_decision = torch.argmax(super_class_output)
        # la indirizzo alla testa scelta da decision
        if super_class is not None:
            item_logits = self.heads[super_class.item()].forward(feature_vector)
        else:
            item_logits = self.heads[super_class_decision.item()].forward(feature_vector)     # caso eval
        item_output = F.softmax(item_logits, dim=1)  # class probability

        return super_class_logits, super_class_output, item_logits, item_output


class Classifier:
    """Classificatore per le 50 classi in esame"""

    def __init__(self, backbone: str, method: str, device: str, ckpt_dir: Path, mapping: dict,
                 weights: Path, pretrained: bool, depth: int):
        """

        :param method:      una tra 'resnet' e resnet18'.
        :param method:      come classificare le varie istanze (naive o mixture of expert).
        :param device:      per decidere se svolgere i conti sulla cpu o sulla gpu.
        :param ckpt_dir:    directory in cui salvare i risultati dei vari esperimetni di train.
        :param mapping:     mapping delle classi.
        :param weights:     percorso dei pesi da caricare.
        :param pretrained:  se usare o meno il modello preallenato come backbone di MoE.
        :param depth:               strati della head.
        """
        self.backbone = backbone
        self.method = method
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.model = None
        self.pretrained = pretrained
        self.mapping = mapping
        if self.method == 'naive':
            self.num_classes = len(self.mapping)
            self.inverse_mapping = {v: k for k, v in self.mapping.items()}
            if self.backbone == 'resnet':
                # carico il modello di resnet50 senza pretrain perchè alleno solo il layer finale
                self.model = resnet50(num_classes=10)
                if self.pretrained:
                    self.load(weights)
                # # congelo i parametri per allenare solo il layer finale
                for p in self.model.parameters():
                    p.requires_grad = False
                # layer finale, a questo giro tanti neuroni quanto il numero di singoli prodotti
                self.model.fc = nn.Linear(2048, self.num_classes)
            elif self.backbone == 'resnet18':
                self.model = resnet18(weights='DEFAULT', progress=True)
                # congelo i parametri tranne quelli degli ultimi 3 blocchi
                # blocks = list(self.model.children())
                # for b in blocks[:-3]:
                #     for p in b.parameters():
                #         p.requires_grad = False
                # layer finale, a questo giro tanti neuroni quanto il numero di singoli prodotti
                self.model.fc = nn.Linear(512, self.num_classes)
            # stampa a schermo la rete
            # summary(self.model, input_size=(1, 3, 224, 224))

        else:
            self.num_super_classes = len(self.mapping)  # numero delle superclassi
            self.super_classes = [i for i in self.mapping]
            self.inverse_submappings = []
            for i in self.super_classes:
                submap = self.mapping[i].copy()
                submap.pop('identifier')
                inverse_submap = {v: k for k, v in submap.items()}
                self.inverse_submappings.append(inverse_submap)
            # lista con il numero di sottoclassi per ogni superclasse (-1 perchè va scartato l'identifier della superclasse)
            self.len_item_classes = [len(self.mapping[key]) - 1 for key in self.mapping]
            # carico il modello pretrainato di resnet50 su imagenet
            self.model = MoE(self.backbone, self.num_super_classes, self.len_item_classes,
                             self.pretrained, weights, self.device, depth)
            # stampa a schermo la rete
            # summary(self.model, input_size=(1, 3, 224, 224), super_class=1)

        self.model.to(self.device)

    def load(self, weights: Path):
        """
        Carica il modello scelto.

        :param weights:     percorso dei pesi da caricare.
        :return:
        """
        model_state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(model_state["model"])
        self.model.to(self.device)

    def forward(self, x: torch.Tensor, super_class):
        """
        Forward step della rete.

        :param x:           esempio di input.
        :param super_class: ground truth della super class.
        :return logits:     output prima della softmax (ci serve per calcolare la loss).
        :return outputs:    output della rete (class probabilities).
        """
        if self.method == 'naive':
            logits = self.model(x)      # output della rete prima di applicare softmax
            outputs = F.softmax(logits, dim=1)      # class probabilities

            return logits, outputs
        else:
            # nel caso stia usando MoE ho già implementato il forward
            return self.model.forward(x, super_class)

    def train_naive_one_epoch(self, dataloader, epoch, optimizer, criterion, writer):
        """
        Allena la rete per un'epoca.

        :param dataloader:  il dataloader del dataset di train.
        :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
        :param optimizer:   per aggiornare i parametri.
        :param criterion:   per la loss.
        :param writer:      per salvare le metriche.
        :return:
        """
        self.model.train()      # modalità train
        optimizer.zero_grad()   # svuoto i gradienti

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='COMPLETED BATCHES')
        epoch_loss = 0.0    # inizializzo la loss
        epoch_correct = 0   # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            images, labels, _ = sample             # non mi interessa della super class
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # output della rete
            logits, outputs = self.forward(images, super_class=None)    # tanto qui la super class non serve
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # loss del batch e backward step
            batch_loss = criterion(logits, labels)
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

    @torch.no_grad()
    def validate_naive(self, dataloader, epoch, criterion, writer):
        """
        Validazione della rete.

        :param dataloader:  il dataloader del dataset di validation.
        :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
        :param criterion:   per la loss.
        :param writer:      per salvare le metriche.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EVAL')
        epoch_loss = 0.0  # inizializzo la loss
        epoch_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_val))
        for sample in progress:
            images, labels, _ = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            logits, outputs = self.forward(images, super_class=None)    # tanto qui la super class non serve
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # loss del batch
            batch_loss = criterion(logits, labels)

            # accumulo le metriche di interesse
            epoch_loss += batch_loss.item()  # avendo usato reduction='sum' nella loss qui sto sommando la loss totale
            batch_correct = torch.sum(batch_decisions == labels)  # risposte corrette per il batch attuale
            epoch_correct += batch_correct.item()  # totale risposte corrette sull'epoca

            postfix = {'batch_mean_loss': batch_loss.item() / batch_cases,
                       'batch_accuracy': (batch_correct.item() / batch_cases) * 100.0}
            progress.set_postfix(postfix)

        epoch_mean_loss = epoch_loss / tot_cases  # loss media sull'epoca
        epoch_accuracy = (epoch_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        writer.add_scalar(f'Loss/Val', epoch_mean_loss, epoch + 1)
        writer.add_scalar(f'Accuracy/Val', epoch_accuracy, epoch + 1)

        return epoch_accuracy

    @torch.no_grad()
    def test_naive(self, dataloader):
        """
        Valuta l'accuratezza della rete sul dataset di test.

        :param dataloader:  il dataloader del dataset di test.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='TEST')
        correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_test))
        inference = []
        for sample in progress:
            images, labels, image_paths = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            _, outputs = self.forward(images, super_class=None)    # tanto qui la super class non serve
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # conto le risposte corrette
            correct += torch.sum(batch_decisions == labels)  # totale risposte corrette

            # debug
            labels = [self.inverse_mapping[i] for i in labels.tolist()]
            batch_decisions = [self.inverse_mapping[i] for i in batch_decisions.tolist()]
            for l, d, p in zip(labels, batch_decisions, image_paths):
                inference.append({'path': p, 'label': l, 'output': d})

        accuracy = (correct / tot_cases) * 100.0  # accuracy sull'epoca (%)

        return accuracy, inference

    def train_moe_one_epoch(self, dataloader, epoch, optimizer, criterion, writer):
        """
        Allena la rete per un'epoca.

        :param dataloader:  il dataloader del dataset di train.
        :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
        :param optimizer:   per aggiornare i parametri.
        :param criterion:   per la loss.
        :param writer:      per salvare le metriche.
        :return:
        """
        self.model.train()      # modalità train
        if self.pretrained:
            self.model.resnet.eval()
        optimizer.zero_grad()   # svuoto i gradienti

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='COMPLETED BATCHES')
        epoch_class_loss = 0.0    # inizializzo la loss delle superclassi
        epoch_class_correct = 0   # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        epoch_item_loss = 0.0  # inizializzo la loss dei prodotti
        epoch_item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            batch_class_decisions = []
            batch_item_decisions = []
            batch_class_loss = 0.0
            batch_item_loss = 0.0

            images, super_class_labels, item_labels, _ = sample
            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di esempi
            # adesso il problema è che per ogni esempio l'architettura della rete cambia, quindi per aggiornare i
            # gradienti non mi viene in mente altro che ciclare sui vari esempi, facendo lo step alla fine del ciclo
            # in modo da preservare la batch_mode
            for image, super_class_label, item_label in zip(images, super_class_labels, item_labels):
                image = torch.unsqueeze(image.to(self.device), dim=0)
                super_class_label = torch.unsqueeze(super_class_label.to(self.device), dim=0)
                item_label = torch.unsqueeze(item_label.to(self.device), dim=0)
                # output della rete
                super_class_logit, super_class_output, item_logit, item_output = self.forward(image, super_class_label)

                # il risultato di softmax viene interpretato con politica winner takes all
                super_class_decision = torch.argmax(super_class_output, dim=1)
                batch_class_decisions.append(super_class_decision)
                item_decision = torch.argmax(item_output, dim=1)
                batch_item_decisions.append(item_decision)

                # loss del batch e backward step

                super_class_loss = criterion(super_class_logit, super_class_label)    # loss sulle classi
                batch_class_loss += super_class_loss
                item_loss = criterion(item_logit, item_label)   # loss sui prodotti
                batch_item_loss += item_loss

                # print('-'*50)
                # print(f'super_class_logit = {super_class_logit}')
                # print(f'super_class_output = {super_class_output}')
                # print(f'super_class_decision = {super_class_decision}')
                # print(f'super_class_label = {super_class_label}')
                # print(f'super_class_logit = {super_class_logit.size()}')
                # print(f'super_class_label = {super_class_label.size()}')
                # print(f'item_label = {item_label}')
                # print(f'item_logit = {item_logit.size()}')
                # print(f'item_label = {item_label.size()}')

            # loss totale, aggiungo enfasi alla class loss perchè determina in cascata la possibilità
            # di classificare corretttamente il prodotto
            total_loss = 2.0 * batch_class_loss + batch_item_loss
            total_loss.backward()

            # aggiorno i pesi
            optimizer.step()
            optimizer.zero_grad()

            # trasformo in tensori le liste in cui ho accumulato le varie loss
            batch_class_decisions = torch.tensor(batch_class_decisions, device=self.device)
            batch_item_decisions = torch.tensor(batch_item_decisions, device=self.device)

            # accumulo le metriche di interesse
            epoch_class_loss += batch_class_loss
            class_bool = batch_class_decisions == super_class_labels.to(self.device)
            batch_class_correct = torch.sum(class_bool)
            epoch_class_correct += batch_class_correct.item()

            epoch_item_loss += batch_item_loss
            # la classificazione del prodotto è corretta se lo era anche quella della super class
            item_bool = batch_item_decisions == item_labels.to(self.device)
            batch_item_correct = torch.sum(torch.logical_and(class_bool, item_bool))
            epoch_item_correct += batch_item_correct.item()

            # print(f'Class decisions: {batch_class_decisions}')
            # print(f'Class labels: {super_class_labels}')
            # print(f'Item decisions: {batch_item_decisions}')
            # print(f'Item labels: {item_labels}')
            # print(f'Class correct: {batch_class_correct}')
            # print(f'Item correct: {batch_item_correct}')
            # print(f'Class accuracy: {(batch_class_correct.item() / batch_cases) * 100.0}')
            # print(f'Item accuracy: {(batch_item_correct.item() / batch_cases) * 100.0}')
            # print('-' * 50)

            postfix = {'batch_mean_class_loss': (batch_class_loss / batch_cases).item(),
                       'batch_class_accuracy': (batch_class_correct.item() / batch_cases) * 100.0,
                       'batch_mean_item_loss': (batch_item_loss / batch_cases).item(),
                       'batch_item_accuracy': (batch_item_correct.item() / batch_cases) * 100.0}
            progress.set_postfix(postfix)

        epoch_mean_class_loss = epoch_class_loss / tot_cases        # loss media sull'epoca
        epoch_mean_item_loss = epoch_item_loss / tot_cases  # loss media sull'epoca
        epoch_class_accuracy = (epoch_class_correct / tot_cases) * 100.0        # accuracy sull'epoca (%)
        epoch_item_accuracy = (epoch_item_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        writer.add_scalar(f'Class Loss/Train', epoch_mean_class_loss, epoch + 1)
        writer.add_scalar(f'Class Accuracy/Train', epoch_class_accuracy, epoch + 1)
        writer.add_scalar(f'Item Loss/Train', epoch_mean_item_loss, epoch + 1)
        writer.add_scalar(f'Item Accuracy/Train', epoch_item_accuracy, epoch + 1)

    @torch.no_grad()
    def validate_moe(self, dataloader, epoch, criterion, writer):
        """
        Validazione della rete.

        :param dataloader:  il dataloader del dataset di validation.
        :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
        :param criterion:   per la loss.
        :param writer:      per salvare le metriche.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EVAL')
        epoch_class_loss = 0.0  # inizializzo la loss delle superclassi
        epoch_class_correct = 0  # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        epoch_item_loss = 0.0  # inizializzo la loss dei prodotti
        epoch_item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            batch_class_decisions = []
            batch_item_decisions = []
            batch_class_loss = 0.0
            batch_item_loss = 0.0

            images, super_class_labels, item_labels, _ = sample
            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di esempi
            # adesso il problema è che per ogni esempio l'architettura della rete cambia, quindi per aggiornare i
            # gradienti non mi viene in mente altro che ciclare sui vari esempi, facendo lo step alla fine del ciclo
            # in modo da preservare la batch_mode
            for image, super_class_label, item_label in zip(images, super_class_labels, item_labels):
                image = torch.unsqueeze(image.to(self.device), dim=0)
                super_class_label = torch.unsqueeze(super_class_label.to(self.device), dim=0)
                item_label = torch.unsqueeze(item_label.to(self.device), dim=0)
                # output della rete
                super_class_logit, super_class_output, item_logit, item_output = self.forward(image, super_class_label)
                # il risultato di softmax viene interpretato con politica winner takes all
                super_class_decision = torch.argmax(super_class_output, dim=1)  # adesso l'argomento è un vettore, non un batch
                batch_class_decisions.append(super_class_decision)
                item_decision = torch.argmax(item_output, dim=1)
                batch_item_decisions.append(item_decision)

                # loss del batch e backward step

                super_class_loss = criterion(super_class_logit, super_class_label)  # loss sulle classi
                batch_class_loss += super_class_loss
                item_loss = criterion(item_logit, item_label)  # loss sui prodotti
                batch_item_loss += item_loss

            # trasformo in tensori le liste in cui ho accumulato le varie loss
            batch_class_decisions = torch.tensor(batch_class_decisions, device=self.device)
            batch_item_decisions = torch.tensor(batch_item_decisions, device=self.device)

            # accumulo le metriche di interesse
            epoch_class_loss += batch_class_loss
            class_bool = batch_class_decisions == super_class_labels.to(self.device)
            batch_class_correct = torch.sum(class_bool)
            epoch_class_correct += batch_class_correct.item()

            epoch_item_loss += batch_item_loss
            # la classificazione del prodotto è corretta se lo era anche quella della super class
            item_bool = batch_item_decisions == item_labels.to(self.device)
            batch_item_correct = torch.sum(torch.logical_and(class_bool, item_bool))
            epoch_item_correct += batch_item_correct.item()

            postfix = {'batch_mean_class_loss': (batch_class_loss / batch_cases).item(),
                       'batch_class_accuracy': (batch_class_correct.item() / batch_cases) * 100.0,
                       'batch_mean_item_loss': (batch_item_loss / batch_cases).item(),
                       'batch_item_accuracy': (batch_item_correct.item() / batch_cases) * 100.0}
            progress.set_postfix(postfix)

        epoch_mean_class_loss = epoch_class_loss / tot_cases  # loss media sull'epoca
        epoch_mean_item_loss = epoch_item_loss / tot_cases  # loss media sull'epoca
        epoch_class_accuracy = (epoch_class_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        epoch_item_accuracy = (epoch_item_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        writer.add_scalar(f'Class Loss/Val', epoch_mean_class_loss, epoch + 1)
        writer.add_scalar(f'Class Accuracy/Val', epoch_class_accuracy, epoch + 1)
        writer.add_scalar(f'Item Loss/Val', epoch_mean_item_loss, epoch + 1)
        writer.add_scalar(f'Item Accuracy/Val', epoch_item_accuracy, epoch + 1)

        return epoch_class_accuracy, epoch_item_accuracy

    @torch.no_grad()
    def test_moe(self, dataloader):
        """
        Valuta l'accuratezza della rete sul dataset di test.

        :param dataloader:  il dataloader del dataset di test.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='TEST')
        class_correct = 0  # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_train))
        inference = []
        for sample in progress:
            class_decisions = []
            item_decisions = []

            images, super_class_labels, item_labels, image_paths = sample
            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di esempi
            # adesso il problema è che per ogni esempio l'architettura della rete cambia, quindi per aggiornare i
            # gradienti non mi viene in mente altro che ciclare sui vari esempi, facendo lo step alla fine del ciclo
            # in modo da preservare la batch_mode
            for image, _, _ in zip(images, super_class_labels, item_labels):
                image = torch.unsqueeze(image.to(self.device), dim=0)
                # # output della rete, a questo giro come superclass prendo la prediction fatta dalla backbone
                super_class_logit, super_class_output, item_logit, item_output = self.forward(image, None)
                # il risultato di softmax viene interpretato con politica winner takes all
                super_class_decision = torch.argmax(super_class_output, dim=1)  # adesso l'argomento è un vettore, non un batch
                class_decisions.append(super_class_decision)
                item_decision = torch.argmax(item_output, dim=1)
                item_decisions.append(item_decision)

            # trasformo in tensori le liste in cui ho accumulato le varie loss
            class_decisions = torch.tensor(class_decisions, device=self.device)
            item_decisions = torch.tensor(item_decisions, device=self.device)

            # accumulo le metriche di interesse
            class_bool = class_decisions == super_class_labels.to(self.device)
            batch_class_correct = torch.sum(class_bool)
            class_correct += batch_class_correct.item()

            # la classificazione del prodotto è corretta se lo era anche quella della super class
            item_bool = (item_decisions == item_labels.to(self.device))
            batch_item_correct = torch.sum(torch.logical_and(class_bool, item_bool))
            item_correct += batch_item_correct.item()

            # # debug
            # super_class_labels = super_class_labels.tolist()
            # class_decisions = class_decisions.tolist()
            # item_labels = item_labels.tolist()
            # item_decisions = item_decisions.tolist()
            # for scl, cd, il, id, p in zip(super_class_labels, class_decisions, item_labels, item_decisions, image_paths):
            #     cd_code = self.super_classes[cd]
            #     scl_code = self.super_classes[scl]
            #     inverse_submap = self.inverse_submappings[cd]
            #     inverse_true_submap = self.inverse_submappings[scl]
            #     inference.append({'path': p, 'super class label': scl_code, 'class output': cd_code,
            #                      'item label': inverse_true_submap[il], 'item output': inverse_submap[id]})

        class_accuracy = (class_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        item_accuracy = (item_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)

        return class_accuracy, item_accuracy, inference

    def train(self, train_loader: DataLoader, val_loader: DataLoader, split: int, epochs: int, lr: float):
        """
        Esegue il train del calssifier.

        :param train_loader:    dataloader del dataset di train.
        :param val_loader:      dataloader del dataset di validation.
        :param split:           indice dello split attuale.
        :param epochs:          numero di epoche per cui eseguire l'allenamento.
        :param lr:              learning rate iniziale.
        :return:
        """
        # così si fa sì che l'optimizer aggiorni solo i parametri non congelati
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr)
        # impostiamo la crossentropy loss con reduction='sum' in modo da poter sommare direttamente le loss di ogni
        # batch e dividerle a fine epoca per ottenere la loss
        criterion = nn.CrossEntropyLoss(reduction='sum')
        ckpt_dir = self.ckpt_dir / f'fold_{split}'
        progress = tqdm(range(epochs), total=epochs, leave=False, desc='COMPLETED EPOCHS')
        # creo un summary writer per salvare le metriche (loss e accuracy)
        writer = SummaryWriter(log_dir=str(ckpt_dir))

        # inizializzo per scegliere il modello migliore
        if self.method == 'naive':
            best_acc = 0.0
            for epoch in progress:
                # alleno la rete su tutti gli esempi del train set (1 epoca)
                self.train_naive_one_epoch(train_loader, epoch, optimizer, criterion, writer)
                # valido il modello attuale sul validation set e ottengo l'accuratezza attuale
                acc_now = self.validate_naive(val_loader, epoch, criterion, writer)
                # scelgo il modello migliore e lo salvo
                if acc_now > best_acc:
                    best_acc = acc_now
                    best_epoch = epoch + 1

                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_epoch,
                        'accuracy': best_acc,
                    }, f=ckpt_dir / 'classifier.pth')

            # restituiso le metriche per stamparle e fare la media sui fold
            best_metrics = {'epoch': best_epoch, 'accuracy': best_acc}
        else:
            best_class_acc = 0.0
            best_item_acc = 0.0
            for epoch in progress:
                # train

                # alleno la rete su tutti gli esempi del train set (1 epoca)
                self.train_moe_one_epoch(train_loader, epoch, optimizer, criterion, writer)
                # valido il modello attuale sul validation set e ottengo l'accuratezza attuale
                class_acc_now, item_acc_now = self.validate_moe(val_loader, epoch, criterion, writer)
                # scelgo il modello migliore e lo salvo
                if class_acc_now >= best_class_acc and item_acc_now > best_item_acc:
                    best_class_acc = class_acc_now
                    best_item_acc = item_acc_now
                    best_epoch = epoch + 1

                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_epoch,
                        'class_accuracy': best_class_acc,
                        'item_accuracy': best_item_acc
                    }, f=ckpt_dir / 'classifier.pth')

            # restituisco le metriche per stamparle e fare la media sui fold
            best_metrics = {'epoch': best_epoch, 'class_accuracy': best_class_acc, 'item_accuracy': best_item_acc}

        return best_metrics


def main(args):
    ROOT = Path(args.root)
    MODE = args.mode
    N_FOLDS = args.n_folds
    DEVICE = args.device
    BACKBONE = args.backbone
    DEPTH = args.depth
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    NOHUP = args.nohup
    METHOD = args.method
    WEIGHTS = Path(args.weights)
    PRETRAINED = args.pretrained
    SEED = args.seed

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert MODE in ['train', 'eval'], '--mode deve essere uno tra "train" e "eval".'
    assert DEVICE in ['cuda', 'cpu'], '--device deve essere uno tra "cuda" e "cpu".'
    assert METHOD in ['naive', 'moe'], 'scegliere --head "naive" se si vuole semplicemente avere tanti output neurons' \
                                       ' quanti sono gli oggetti oppure "moe" per usare un ensemble di classificatori' \
                                       ' specifici ciascuno per ogni categoria merceologica.'

    train_transforms = Compose([ToTensor(),
                                RandomAffine(45, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=255),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                          std=torch.tensor([0.229, 0.224, 0.225]))])

    val_transforms = Compose([ToTensor(),
                              Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                        std=torch.tensor([0.229, 0.224, 0.225]))])

    # train mode
    if MODE == 'train':
        if not NOHUP:
            # creo una cartella dove salverò l'andamento dei vari allenamenti, serve solo se sto trainando
            CHECKPOINT_DIR = Path('runs/instance_level')
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # creo la directory se già non esiste
            past_experiments = len(
                os.listdir(CHECKPOINT_DIR))  # la prima si chiamerà run_0, la seconda run_1 e così via
            actual_dir = CHECKPOINT_DIR / f'run_{past_experiments}'  # qui salvo i risultati degli esperimenti
            for i in range(N_FOLDS):
                # makedirs crea tutte le cartelle intermedie che ancora non esistono specificate nel path
                # exists_ok fa si che se una cartella esiste già non c'è un errore
                os.makedirs(actual_dir / f'fold_{i}', exist_ok=True)  # qui salvo i risultati del singolo split
            with open(actual_dir / 'info.txt', 'w') as f:
                json.dump(arguments.__dict__, f, indent=2)
        else:
            actual_dir = CHECKPOINT_DIR
        # creo una cartella dove salverò l'andamento dei vari allenamenti, serve solo se sto trainando
        splits = []     # qui salvo gli n_folds dataset che sono i singoli split
        best_results = []
        for i in tqdm(range(N_FOLDS), total=N_FOLDS, desc='Creo gli split del dataset.'):
            splits.append(MyDataset(ROOT, N_FOLDS, i, mode=MODE, transforms=val_transforms, method=METHOD, seed=SEED))

        for i, split in tqdm(enumerate(splits), total=N_FOLDS, desc='COMPLETED FOLDS'):
            # ciclicamente uso uno split come val, reimposto le transforms a val_transforms nel caso fossero state
            # cambiate in precedenza
            val_ds = split  # split è il dataset che sto usando come validation
            class_mapping = val_ds.mapping
            val_ds.set_transforms(val_transforms)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # gli altri split li uso per il train, mettendoli in una lista e passandoli a Concat, settando prima le
            # train_transform
            train_datasets = [j for j in splits if j is not split]  # come train uso gli altri, unendoli con Concat
            for d in train_datasets:
                d.set_transforms(train_transforms)
            train_ds = Concat(train_datasets)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

            cls = Classifier(BACKBONE, METHOD, DEVICE, actual_dir, class_mapping, WEIGHTS, PRETRAINED, DEPTH)
            train_result = cls.train(train_loader, val_loader, i, EPOCHS, LR)      # alleno
            best_results.append(train_result)

        if METHOD == 'naive':
            for i, r in enumerate(best_results):
                print(f'Fold {i + 1}: miglior accuratezza raggiunta dopo {r["epoch"]} epoche pari al {r["accuracy"]}%.')
            accuracies = [r["accuracy"] for r in best_results]  # elenco le best_accuracy di ogni fold per la media
            mean_accuracy = np.mean(accuracies)
            print(f'Accuracy media: {mean_accuracy}%.')
        else:
            for i, r in enumerate(best_results):
                print(f'Fold {i + 1}: miglior accuratezza raggiunta dopo {r["epoch"]} epoche. Class accuracy pari al '
                      f'{r["class_accuracy"]}%, item accuracy pari al {r["item_accuracy"]}%.')
            class_accuracies = [r["class_accuracy"] for r in best_results]
            mean_class_accuracy = np.mean(class_accuracies)
            print(f'Class accuracy media: {mean_class_accuracy}%.')
            item_accuracies = [r["item_accuracy"] for r in best_results]
            mean_item_accuracy = np.mean(item_accuracies)
            print(f'Item accuracy media: {mean_item_accuracy}%.')

    else:
        # a questo giro deve essere il percorso alla cartella dell'esperimento
        experiment_dir = CHECKPOINT_DIR
        # per creare il dataset passo il parametro split ma non serve (__init__ lo setta a n_folds)
        test_ds = MyDataset(ROOT, N_FOLDS, split=0, mode=MODE, transforms=val_transforms, method=METHOD, seed=SEED)
        class_mapping = test_ds.mapping
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        cls = Classifier(BACKBONE, METHOD, DEVICE, experiment_dir, class_mapping, WEIGHTS, pretrained=False, depth=DEPTH)  # inizializzo il classificatore
        naive_acc, moe_class_acc, moe_item_acc = [], [], []
        for i in range(N_FOLDS):
            weights = experiment_dir / f'fold_{i}' / 'classifier.pth'
            cls.load(weights)
            if METHOD == 'naive':
                test_accuracy, inference = cls.test_naive(test_loader)
                naive_acc.append(test_accuracy)
                with open(experiment_dir / f'fold_{i}' / 'inference.json', 'w') as f:
                    json.dump(inference, f)
                print(f'Accuracy sui dati di test durante il fold {i + 1}: {test_accuracy}%.')
            else:
                class_accuracy, item_accuracy, inference = cls.test_moe(test_loader)
                moe_class_acc.append(class_accuracy)
                moe_item_acc.append(item_accuracy)
                # with open(experiment_dir / f'fold_{i}' / 'inference.json', 'w') as f:
                #     json.dump(inference, f)
                print(f'Class accuracy sui dati di test durante il fold {i + 1}: {class_accuracy}%.')
                print(f'Item accuracy sui dati di test durante il fold {i + 1}: {item_accuracy}%.')
        if METHOD == 'naive':
            print(f'Naive accuracy media: {torch.mean(torch.tensor(naive_acc))}%.')
        else:
            print(f'Class cccuracy media: {torch.mean(torch.tensor(moe_class_acc))}%.')
            print(f'Item accuracy media: {torch.mean(torch.tensor(moe_item_acc))}%.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train del classificatore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='/home/deepmammo/tommaso/prove/subset_10/',
                        help='Root del dataset. (C, I, F)')
    parser.add_argument('--n-folds', type=int, default=3, help='Numero di fold per la cross validation. (C, I, F)')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Scegliere tra "train" e "eval" in base a quale modalità si desidera. (C, I, F)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Scegliere se usare la gpu ("cuda") o la "cpu". (C, I, F)')
    parser.add_argument('-b', '--backbone', type=str, default='resnet',
                        help='Scegliere se utilizzare una semplice "cnn", "resnet" (50), "resnet18", "resnet10" '
                             'o "resnet12" come features extractor. (C, I, F)')
    parser.add_argument('--depth', type=int, default=1, help='Strati della head. (I)')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Epoche per eseguire il train. (C, I, F)')
    parser.add_argument('--batch-size', type=int, default=16, help='Numero di esempi in ogni batch. (C, I)')
    parser.add_argument('--num-workers', type=int, default=3, help='Numero di worker. (C, I, F)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. (C, I, F)')
    parser.add_argument('--seed', type=int, default=123, help='Per riproducibilità. (C, I, F)')
    parser.add_argument('--weights', type=str, default='classifier.pth',
                        help='Percorso dei pesi da usare per il classificatore. (C, I, F)')
    parser.add_argument('--method', type=str, default='moe',
                        help='Scegliere se usare un approccio "naive" o "moe" se siamo in "instance_level", '
                             '"proto", "match" o "rel" se siamo in fsl. (I, F)')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Ha senso solo durante il train se vogliamo allenare solo le teste ed utilizzare un '
                             'modello già allenato per fare la classificazione iniziale. (I)')
    parser.add_argument('--episodes', type=int, default=500, help='Numero di episodi per ogni epoca. (F)')
    parser.add_argument('--val-episodes', type=int, default=100,
                        help='Numero di episodi per ogni step di validazione. (F)')
    parser.add_argument('--test-episodes', type=int, default=1000, help='Numero di episodi per il test. (F)')
    parser.add_argument('--n-way', type=int, default=5, help='Numero di classi per ogni episodio. (F)')
    parser.add_argument('--k-shot', type=int, default=1, help='Numero di esempi per ogni classe nel support set. (F)')
    parser.add_argument('--n-query', type=int, default=4, help='Numero di esempi per ogni classe nel query set. (F)')
    # -----------QUESTI-VENGONO-PASSATI-SOLO-DA-NOHUP-E-NON-VANNO-USATI-------------------------------------------------
    parser.add_argument('--checkpoint-dir', type=str, default='runs/classifier',
                        help='Cartella dove salvare i risultati dei vari esperimenti. (C, I, F)')
    parser.add_argument('--nohup', type=bool, default=False, help='Se lancio da nohup passo true. (C, I, F)')

    arguments = parser.parse_args()
    main(arguments)
