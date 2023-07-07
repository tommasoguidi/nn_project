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
from torchvision import transforms
from torchvision.models.resnet import resnet50, ResNet
from torchvision.models._utils import _ovewrite_named_param
from torchinfo import summary


class MyDataset(Dataset):
    """
    Crea gli split del dataset. Per eseguire la cross validation, uno di questi dovrà essere usato come validation,
    gli altri devono essere uniti per ottenere il train usando la classe Concat.
    """
    def __init__(self, path: Path, n_folds: int, split: int, mode: str, transforms: transforms.Compose):
        """
        Crea il dataset a partire dal percorso indicato.

        :param path:        percorso della cartella contenente il dataset
        :param n_folds:     numero di fold per crossvalidation
        :param split:       indice per identificare quale split stiamo usando
        :param mode:        per dire se stiamo facendo 'train' o 'eval'
        :param transforms:  le trasformazioni da applicare all'immagine
        """
        self.path = path
        self.img_dir = path / 'images'
        self.annos = pd.read_csv(path / 'subset_10_images.csv', index_col='image_id')  # leggo il csv con pandas
        self.n_folds = n_folds
        self.split = split
        self.mode = mode
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
        self.all_classes = self.annos['item_id'].unique().tolist()   # tutte le classi del dataset
        self.all_super_classes = sorted(self.annos['product_type'].unique().tolist())     # classi del caso semplice
        self.mapping = {}
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
        item_label = torch.tensor(self.mapping[super_class_label][item_label], dtype=torch.long)
        super_class_label = torch.tensor(self.mapping[super_class_label]['identifier'], dtype=torch.long)

        return image, super_class_label, item_label

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
    def __init__(self, in_features: int, classes: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features, classes)

    def forward(self, x):
        logits = self.layer1(x)  # output della rete prima di applicare softmax
        return logits


class MyResNet(nn.Module):
    """Riscrivo il forward della ResNet originaria per prelevare il vettore delle features"""
    def __init__(self, num_super_classes):
        super().__init__()
        # carico la resnet
        self.resnet = resnet50(weights='DEFAULT', progress=True)
        # congelo i parametri tranne quelli dell'ultimo bottleneck
        blocks = list(self.resnet.children())
        for b in blocks[:-3]:
            for p in b.parameters():
                p.requires_grad = False
        # modifico il linear layer per la classificazione della super classe
        self.resnet.fc = nn.Linear(2048, num_super_classes)

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
    def __init__(self, num_super_classes: int, len_item_classes: list):
        """

        :param num_super_classes:   numero delle super classi.
        :param len_item_classes:    lista con il numero di prodotti per ciascuna super classe
                                    -> len_item_classes[i] è il numero di prodotti della classe i.
        """
        super().__init__()
        self.num_super_classes = num_super_classes
        self.len_item_classes = len_item_classes
        # metto la mia resnet con forward modificato
        self.resnet = MyResNet(num_super_classes=self.num_super_classes)
        # creo un'istanza della classe Head() per ogni super classe e la aggiungo alla ModuleList
        # la i-esima istanza ha come in_features la dimensione delle feature uscenti dalla resnet dopo flatten() (2048)
        # e come classes il numero dei prodotti appartenenti alla i-esima super class
        self.heads = nn.ModuleList([Head(2048, self.len_item_classes[i]) for i in range(self.num_super_classes)])

    def forward(self, x):
        # il metodo forward() di resnet è stato modificato per ritornare anche il feature vector
        super_class_logits, feature_vectors = self.resnet.forward(x)
        super_class_outputs = F.softmax(super_class_logits, dim=1)  # class probabilities
        super_class_decision = torch.argmax(super_class_outputs, dim=1)  # questo è un tensore di dimensione batch_size
        # ciclo su ogni risultato e lo indirizzo alla testa giusta
        all_item_logits = ()
        all_item_outputs = ()
        for i, decision in enumerate(super_class_decision):
            # prendo una per una le encoding delle immagini
            feature_encoding = feature_vectors[i]
            # la indirizzo alla testa scelta da decision
            item_logits = self.heads[decision.item()].forward(feature_encoding)
            all_item_logits += (item_logits,)
            item_outputs = F.softmax(item_logits)  # class probabilities
            all_item_outputs += (item_outputs,)

        item_logits = torch.vstack(all_item_logits)
        print(item_logits.size())
        item_outputs = torch.vstack(all_item_outputs)

        return super_class_logits, super_class_outputs, item_logits, item_outputs


class Classifier:
    """Classificatore per le 50 classi in esame"""

    def __init__(self, method: str, device: str, ckpt_dir: Path, mapping: dict, weights: Path):
        """

        :param method:      come classificare le varie istanze (naive o mixture of expert).
        :param device:      per decidere se svolgere i conti sulla cpu o sulla gpu.
        :param ckpt_dir:    directory in cui salvare i risultati dei vari esperimetni di train.
        :param mapping:     mapping delle classi.
        :param weights:     percorso dei pesi da caricare.
        """
        self.method = method
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.model = None
        self.num_super_classes = len(mapping)   # numero delle superclassi
        # lista con il numero di sottoclassi per ogni superclasse (-1 perchè va scartato l'identifier della superclasse)
        self.len_item_classes = [len(mapping[key]) - 1 for key in mapping]

        if self.method == 'naive':
            # carico il modello di resnet50 senza pretrain
            self.model = resnet50()
            # prima di caricare i pesi del classificatore allenato lo devo modificare come era stato fatto
            # in precedenza, quindi modifico il layer finale della resnet
            self.model.fc = nn.Linear(2048, self.num_super_classes)
            self.load(weights)
            # # congelo i parametri per allenare solo il layer finale
            # for p in self.model.parameters():
            #     p.requires_grad = False
            # congelo i parametri tranne quelli degli ultimi 3 blocchi
            blocks = list(self.model.children())
            for b in blocks[:-3]:
                for p in b.parameters():
                    p.requires_grad = False
            # layer finale, a questo giro tanti neuroni quanto il numero di singoli prodotti
            self.model.fc = nn.Linear(2048, sum(self.len_item_classes))

        else:
            # carico il modello pretrainato di resnet50 su imagenet
            self.model = MoE(self.num_super_classes, self.len_item_classes)

        # # stampa a schermo la rete
        # summary(self.model, input_size=(1, 3, 224, 224))
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

    def forward(self, x: torch.Tensor):
        """
        Forward step della rete.

        :param x:           esempio di input.
        :return logits:     output prima della softmax (ci serve per calcolare la loss).
        :return outputs:    output della rete (class probabilities).
        """
        if self.method == 'naive':
            logits = self.model(x)      # output della rete prima di applicare softmax
            outputs = F.softmax(logits, dim=1)      # class probabilities

            return logits, outputs
        else:
            # nel caso stia usando MoE ho già implementato il forward
            return self.model.forward(x)

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
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EPOCH')
        epoch_loss = 0.0    # inizializzo la loss
        epoch_correct = 0   # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            images, _, labels = sample             # non mi interessa della super class
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # output della rete
            logits, outputs = self.forward(images)
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
                       'batch_accuracy': batch_correct.item()/batch_cases}
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
            images, _, labels = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            logits, outputs = self.forward(images)
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
        for sample in progress:
            images, _, labels = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            _, outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # conto le risposte corrette
            correct += torch.sum(batch_decisions == labels)  # totale risposte corrette

        accuracy = (correct / tot_cases) * 100.0  # accuracy sull'epoca (%)

        return accuracy

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
        optimizer.zero_grad()   # svuoto i gradienti

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EPOCH')
        epoch_class_loss = 0.0    # inizializzo la loss delle superclassi
        epoch_class_correct = 0   # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        epoch_item_loss = 0.0  # inizializzo la loss dei prodotti
        epoch_item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            images, super_class_labels, item_labels = sample
            images = images.to(self.device)
            super_class_labels = super_class_labels.to(self.device)
            item_labels = item_labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # output della rete
            super_class_logits, super_class_outputs, item_logits, item_outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_class_decisions = torch.argmax(super_class_outputs, dim=1)
            batch_item_decisions = torch.argmax(item_outputs, dim=1)

            # loss del batch e backward step
            batch_class_loss = criterion(super_class_logits, super_class_labels)    # loss sulle classi
            batch_item_loss = criterion(item_logits, item_labels)   # loss sui prodotti
            # loss totale, aggiungo enfasi alla class loss perchè determina in cascata la possibilità di classificare
            # corretttamente il prodotto
            batch_total_loss = 2.0 * batch_class_loss + batch_item_loss
            batch_total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # accumulo le metriche di interesse
            epoch_class_loss += batch_class_loss.item()
            batch_class_correct = torch.sum(batch_class_decisions == super_class_labels)
            epoch_class_correct += batch_class_correct.item()

            epoch_item_loss += batch_item_loss.item()
            # la classificazione del prodotto è corretta se lo era anche quella della super class
            batch_item_correct = torch.sum((batch_class_decisions == super_class_labels) and (batch_item_decisions == item_labels))
            epoch_item_correct += batch_item_correct.item()

            postfix = {'batch_mean_class_loss': batch_class_loss.item()/batch_cases,
                       'batch_class_accuracy': batch_class_correct.item()/batch_cases,
                       'batch_mean_item_loss': batch_item_loss.item() / batch_cases,
                       'batch_item_accuracy': batch_item_correct.item() / batch_cases}
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
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EPOCH')
        epoch_class_loss = 0.0  # inizializzo la loss delle superclassi
        epoch_class_correct = 0  # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        epoch_item_loss = 0.0  # inizializzo la loss dei prodotti
        epoch_item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            images, super_class_labels, item_labels = sample
            images = images.to(self.device)
            super_class_labels = super_class_labels.to(self.device)
            item_labels = item_labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # output della rete
            super_class_logits, super_class_outputs, item_logits, item_outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_class_decisions = torch.argmax(super_class_outputs, dim=1)
            batch_item_decisions = torch.argmax(item_outputs, dim=1)

            # loss del batch e backward step
            batch_class_loss = criterion(super_class_logits, super_class_labels)  # loss sulle classi
            batch_item_loss = criterion(item_logits, item_labels)  # loss sui prodotti

            # accumulo le metriche di interesse
            epoch_class_loss += batch_class_loss.item()
            batch_class_correct = torch.sum(batch_class_decisions == super_class_labels)
            epoch_class_correct += batch_class_correct.item()

            epoch_item_loss += batch_item_loss.item()
            batch_item_correct = torch.sum((batch_class_decisions == super_class_labels) and (batch_item_decisions == item_labels))
            epoch_item_correct += batch_item_correct.item()

            postfix = {'batch_mean_class_loss': batch_class_loss.item() / batch_cases,
                       'batch_class_accuracy': batch_class_correct.item() / batch_cases,
                       'batch_mean_item_loss': batch_item_loss.item() / batch_cases,
                       'batch_item_accuracy': batch_item_correct.item() / batch_cases}
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
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='EPOCH')
        class_correct = 0  # prediction corrette della rete per calcolare l'accuracy sulle superclassi
        item_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy sui prodotti
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            images, super_class_labels, item_labels = sample
            images = images.to(self.device)
            super_class_labels = super_class_labels.to(self.device)
            item_labels = item_labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            super_class_logits, super_class_outputs, item_logits, item_outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_class_decisions = torch.argmax(super_class_outputs, dim=1)
            batch_item_decisions = torch.argmax(item_outputs, dim=1)

            # conto le risposte corrette
            class_correct += torch.sum(batch_class_decisions == super_class_labels)  # totale risposte corrette
            item_correct += torch.sum((batch_class_decisions == super_class_labels) and (batch_item_decisions == item_labels))
        class_accuracy = (class_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        item_accuracy = (item_correct / tot_cases) * 100.0

        return class_accuracy, item_accuracy

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
        # impostiamo la crossentropy loss con reduction='sum' in modo da poter sommare direttamente le loss di ogni
        # batch e dividerle a fine epoca per ottenere la loss
        criterion = nn.CrossEntropyLoss(reduction='sum')
        ckpt_dir = self.ckpt_dir / f'fold_{split}'
        progress = tqdm(range(epochs), total=epochs, leave=False, desc='FOLD')
        # creo un summary writer per salvare le metriche (loss e accuracy)
        writer = SummaryWriter(log_dir=str(ckpt_dir))

        # inizializzo per scegliere il modello migliore
        if self.method == 'naive':
            best_acc = 0.0
            for epoch in progress:
                # train

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
                if class_acc_now > best_class_acc and item_acc_now > best_item_acc:
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
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    METHOD = args.method
    WEIGHTS = Path(args.weights)

    assert MODE in ['train', 'eval'], '--mode deve essere uno tra "train" e "eval".'
    assert DEVICE in ['cuda', 'gpu'], '--device deve essere uno tra "cuda" e "gpu".'
    assert METHOD in ['naive', 'moe'], 'scegliere --head "naive" se si vuole semplicemente avere tanti output neurons' \
                                       ' quanti sono gli oggetti oppure "moe" per usare un ensemble di classificatori' \
                                       ' specifici ciascuno per ogni categoria merceologica.'

    train_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomAffine(45, translate=(0.1, 0.1),
                                                                   scale=(0.8, 1.2), fill=255),
                                           transforms.RandomHorizontalFlip(p=0.5)])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    # train mode
    if MODE == 'train':
        # creo una cartella dove salverò l'andamento dei vari allenamenti, serve solo se sto trainando
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # creo la directory se già non esiste
        past_experiments = len(os.listdir(CHECKPOINT_DIR))  # la prima si chiamerà run_0, la seconda run_1 e così via
        actual_dir = CHECKPOINT_DIR / f'run_{past_experiments}'     # qui salvo i risultati degli esperimenti
        for i in range(N_FOLDS):
            # makedirs crea tutte le cartelle intermedie che ancora non esistono specificate nel path
            # exists_ok fa si che se una cartella esiste già non c'è un errore
            os.makedirs(actual_dir / f'fold_{i}', exist_ok=True)      # qui salvo i risultati del singolo split

        splits = []     # qui salvo gli n_folds dataset che sono i singoli split
        best_results = []
        for i in tqdm(range(N_FOLDS), total=N_FOLDS, desc='Creo gli split del dataset.'):
            splits.append(MyDataset(ROOT, N_FOLDS, i, mode=MODE, transforms=val_transforms))

        for i, split in tqdm(enumerate(splits), total=N_FOLDS, desc=f'{N_FOLDS}-fold cross validation...'):
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

            cls = Classifier(METHOD, DEVICE, actual_dir, class_mapping, WEIGHTS)      # inizializzo il classificatore
            train_result = cls.train(train_loader, val_loader, i, EPOCHS, LR)      # alleno
            best_results.append(train_result)

        for i, r in enumerate(best_results):
            print(f'Fold {i+1}: miglior accuratezza raggiunta dopo {r["epoch"]} epoche pari al {r["accuracy"]}%.')
        if METHOD == 'naive':
            accuracies = [r["accuracy"] for r in best_results]  # elenco le best_accuracy di ogni fold per la media
            mean_accuracy = np.mean(accuracies)
            print(f'Accuracy media: {mean_accuracy}')
        else:
            class_accuracies = [r["class_accuracy"] for r in best_results]
            mean_class_accuracy = np.mean(class_accuracies)
            print(f'Class accuracy media: {mean_class_accuracy}')
            item_accuracies = [r["item_accuracy"] for r in best_results]
            mean_item_accuracy = np.mean(item_accuracies)
            print(f'Item accuracy media: {mean_item_accuracy}')

    else:
        # a questo giro deve essere il percorso completo alla cartella in cui sono stati salvati i progressi
        # del modello prescelto
        actual_dir = CHECKPOINT_DIR
        # per creare il dataset non passo il parametro split perchè non serve (__init__ lo setta a n_folds)
        test_ds = MyDataset(ROOT, N_FOLDS, mode=MODE, transforms=val_transforms)
        class_mapping = test_ds.mapping
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        cls = Classifier(METHOD, DEVICE, actual_dir, class_mapping, WEIGHTS)  # inizializzo il classificatore
        cls.load(WEIGHTS)

        if METHOD == 'naive':
            test_accuracy = cls.test_naive(test_loader)
            print(f'Accuracy sui dati di test: {test_accuracy}')
        else:
            class_accuracy, item_accuracy = cls.test_moe(test_loader)
            print(f'Class accuracy sui dati di test: {class_accuracy}')
            print(f'Item accuracy sui dati di test: {item_accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train del classificatore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\subset_10',
                        help='Root del dataset.')
    parser.add_argument('--n-folds', type=int, default=3, help='Numero di fold per la cross validation')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Scegliere tra "train" e "eval" in base a quale modalità si desidera.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Scegliere se usare la gpu ("cuda") o la "cpu".')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Epoche per eseguire il train')
    parser.add_argument('--batch-size', type=int, default=16, help='Numero di esempi in ogni batch.')
    parser.add_argument('--num-workers', type=int, default=3, help='Numero di worker.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--checkpoint_dir', type=str, default='runs/instance_level',
                        help='Cartella dove salvare i risultati dei vari esperimenti. Se --mode == "train" specificare'
                             ' la cartella madre che contiene tutte le annotazioni sugli esperimenti; se --mode =='
                             ' "eval" indicare la cartella dello specifico esperimento che si vuole valutare.')
    parser.add_argument('--method', type=str, default='moe',
                        help='Scegliere se usare un approccio "naive" (#neuroni_out == #classi) o "moe"'
                             ' (mixture of experts). Il metodo naive carica il classificatore ottenuto al passo'
                             ' precedente e ne allena gli strati finali usando tanti neuroni di output quanti'
                             ' sono i singoli prodotti; il metodo moe usa una ensemble di classificatori, ciascuno'
                             ' specializzato su una determinata categoria merceologica, riallenando al tempo stesso'
                             ' anche la backbone a partire dai pesi di imagenet.')
    parser.add_argument('--weights', type=str, default='classifier.pth',
                        help='Percorso dei pesi da usare per il feature extractor.')

    arguments = parser.parse_args()
    main(arguments)
