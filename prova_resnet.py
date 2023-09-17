import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomAffine, Normalize
from torchvision.models import resnet50, resnet18
from torchinfo import summary


class MyDataset(Dataset):
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
            self.split = self.n_folds   # quando testo voglio usare il set di hold out che ho escluso dagli split
        self.transforms = transforms
        self.all_ids = self.annos.index.tolist()    # lista di tutti gli image_id
        self.all_classes = sorted(self.annos['product_type'].unique().tolist())     # tutte le classi del dataset
        self.mapping = {i: j for j, i in enumerate(self.all_classes)}       # mapping classe_i -> j (intero)
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

        # a questo punto del progetto l'interesse è quello di allenare un classificatore in grado di riconoscere la
        # categoria merceologica cui l'oggetto appartiene. la label quindi sarà il valore 'product_type' che viene
        # mappato al suo valore intero di riferimento e trasformato in un tensore
        label = self.annos.loc[image_id, 'product_type']
        label = torch.tensor(self.mapping[label], dtype=torch.long)

        return image, label

    def _get_ids_in_split(self):
        """
        Restituisce la lista dei percorsi delle immagini nello split attuale. In particolare, la funzione suddivide
        il dataset completo in n_folds+1 parti (bilanciate) così da poter fare crossvalidation usando stratified
        k-folding e al tempo stesso tenere da parte un hold out set per fare test
        """
        ids_in_split = []
        for _class in self.all_classes:
            # metto in una lista tutti gli image_id di una determinata classe
            ids_in_class = self.annos.index[self.annos['product_type'] == _class].tolist()
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
    dataset e verrà utilizzata per ottenere i dataset di train.
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


class MyResNet(nn.Module):
    """Riscrivo il forward della ResNet originaria per prelevare il vettore delle features"""
    def __init__(self):
        super().__init__()
        self.device = 'cuda'

        self.resnet = resnet50()
        # modifico il final layer per poter caricare i miei pesi
        self.resnet.fc = nn.Linear(2048, 10)
        model_state = torch.load('runs/classifier/run_7/fold_0/classifier.pth', map_location=self.device)
        self.resnet.load_state_dict(model_state["model"])

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


class Head(nn.Module):
    """Questa è la classe base delle classification heads usate nel caso 'moe'."""
    def __init__(self, in_features: int, classes: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 4096)
        self.layer2 = nn.Linear(4096, 2048)
        self.layer3 = nn.Linear(2048, classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.dropout(x)
        logits = self.layer3(x)  # output della rete prima di applicare softmax
        return logits


class MoE(nn.Module):
    """
    Il modulo è composto dalla resnet preallenata su imagenet e una lista di classification heads, una per ciascuna
    classe.
    """
    def __init__(self):
        """
        """
        super().__init__()
        # metto la mia resnet con forward modificato
        self.resnet = MyResNet()
        # creo un'istanza della classe Head() per ogni super classe e la aggiungo alla ModuleList
        # la i-esima istanza ha come in_features la dimensione delle feature uscenti dalla resnet dopo flatten() (2048)
        # e come classes il numero dei prodotti appartenenti alla i-esima super class
        self.heads = nn.ModuleList([Head(2048, 100) for i in range(10)])

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

        return super_class_logits, super_class_output


class Classifier:
    """Classificatore per le 50 classi in esame"""

    def __init__(self, backbone: str, mapping: dict):
        """
        Classe che consente di creare, allenare e testare il modello responsabile della classificazione.

        :param backbone:    la rete alla base del classificatore (cnn base o resnet).
        :param device:      per decidere se svolgere i conti sulla cpu o sulla gpu.
        :param ckpt_dir:    directory in cui salvare i risultati dei vari esperimenti di train.
        :param mapping:     mapping delle classi.
        """
        self.device = 'cuda'
        self.model = None
        self.outputs = len(mapping)   # quante sono le classi nel nostro caso
        self.backbone = backbone

        if self.backbone == 'myresnet':
            self.model = MoE()

        elif self.backbone == 'resnet':
            # carico il modello pretrainato di resnet50 su imagenet
            self.model = resnet50(weights='DEFAULT', progress=True)
            # congelo i parametri tranne quelli degli ultimi 3 blocchi
            blocks = list(self.model.children())
            for b in blocks[:-3]:
                for p in b.parameters():
                    p.requires_grad = False
            # layer finale
            self.model.fc = nn.Linear(2048, 10)

            model_state = torch.load('runs/classifier/run_7/fold_0/classifier.pth', map_location=self.device)
            self.model.load_state_dict(model_state["model"])

        self.model.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward step della rete.

        :param x:           esempio di input.
        :return logits:     output prima della softmax (ci serve per calcolare la loss).
        :return outputs:    output della rete (class probabilities).
        """

        if self.backbone == 'resnet':
            logits = self.model(x)      # output della rete prima di applicare softmax
            outputs = F.softmax(logits, dim=1)      # class probabilities
        else:
            logits, outputs = self.model(x)

        return logits, outputs

    @torch.no_grad()
    def test(self, dataloader):
        """
        Valuta l'accuratezza della rete sul dataset di test.

        :param dataloader:  il dataloader del dataset di test.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False, desc='TEST')
        correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy
        top3_count = 0
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_test))
        for sample in progress:
            images, labels = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample

            # outputs della rete
            _, outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)
            top3_decisions = torch.topk(outputs, k=3, dim=1).indices    # stesso motivo per cui uso argmax

            # conto le risposte corrette
            correct += torch.sum(batch_decisions == labels)  # totale risposte corrette
            # faccio la top3 accuracy
            for gt, top3 in zip(labels, top3_decisions):
                top3_count += int(torch.isin(gt, top3).item())

        accuracy = (correct / tot_cases) * 100.0  # accuracy sull'epoca (%)
        top3_acc = (top3_count / tot_cases) * 100.0  # top3_accuracy sull'epoca (%)

        return accuracy, top3_acc

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
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_train))
        for sample in progress:
            class_decisions = []

            images, super_class_labels = sample
            batch_cases = images.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di esempi
            # adesso il problema è che per ogni esempio l'architettura della rete cambia, quindi per aggiornare i
            # gradienti non mi viene in mente altro che ciclare sui vari esempi, facendo lo step alla fine del ciclo
            # in modo da preservare la batch_mode
            for image, _ in zip(images, super_class_labels):
                image = torch.unsqueeze(image.to(self.device), dim=0)
                # output della rete, a questo giro come superclass prendo la prediction fatta dalla backbone
                super_class_logit, super_class_output = self.forward(image)
                # il risultato di softmax viene interpretato con politica winner takes all
                super_class_decision = torch.argmax(super_class_output)  # adesso l'argomento è un vettore, non un batch
                class_decisions.append(super_class_decision)

            # trasformo in tensori le liste in cui ho accumulato le varie loss
            class_decisions = torch.tensor(class_decisions, device=self.device)

            # accumulo le metriche di interesse
            class_bool = class_decisions == super_class_labels.to(self.device)
            batch_class_correct = torch.sum(class_bool)
            class_correct += batch_class_correct.item()

        class_accuracy = (class_correct / tot_cases) * 100.0  # accuracy sull'epoca (%)

        return class_accuracy


def main(args):
    ROOT = Path(args.root)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SEED = args.seed

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    val_transforms = Compose([ToTensor(),
                              Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                        std=torch.tensor([0.229, 0.224, 0.225]))])

    test_ds = MyDataset(ROOT, 3, split=0, mode='eval', transforms=val_transforms, seed=SEED)
    class_mapping = test_ds.mapping
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    for i in ['resnet', 'myresnet']:
        cls = Classifier(i, class_mapping)  # inizializzo il classificatore

        test_accuracy = cls.test_moe(test_loader)
        print(f'Risultati ottenuti da {i}:')
        print(f'Accuracy sui dati di test: {test_accuracy}%')
        # print(f'Top3-Accuracy sui dati di test: {top3_accuracy}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train del classificatore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\subset_10',
                        help='Root del dataset.')
    parser.add_argument('--batch-size', type=int, default=16, help='Numero di esempi in ogni batch.')
    parser.add_argument('--num-workers', type=int, default=3, help='Numero di worker.')
    parser.add_argument('--seed', type=int, default=123, help='Per riproducibilità.')

    arguments = parser.parse_args()
    main(arguments)
