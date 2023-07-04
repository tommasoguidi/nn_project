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
from torchvision.models import resnet50
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
        self.annos = pd.read_csv(path / 'subset_images.csv', index_col='image_id')  # leggo il csv con pandas
        self.n_folds = n_folds
        self.split = split
        self.mode = mode
        # in eval mode vogliamo usare il test set, che è stato ottenuto dalla coda del dataset totale (maggiori info
        # sotto _get_ids_in_split())
        if self.mode == 'eval':
            self.split = self.n_folds
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
            # metto in una lista tutte gli image_id di una determinata classe
            ids_in_class = self.annos.index[self.annos['product_type'] == _class].tolist()
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


class Classifier:
    """Classificatore per le 50 classi in esame"""

    def __init__(self, backbone: str, device: str, ckpt_dir: Path):
        """

        :param backbone:    la rete alla base del classificatore (cnn base o resnet).
        :param device:      per decidere se svolgere i conti sulla cpu o sulla gpu.
        :param ckpt_dir:    directory in cui salvare i risultati dei vari esperimetni di train.
        """
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.model = None
        self.outputs = 50   # quante sono le classi nel nostro caso

        if backbone == 'cnn':
            # implementazione di una cnn standard
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3),
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(inplace=True),
                nn.Linear(4096, self.outputs)
            )

        elif backbone == 'resnet':
            # carico il modello pretrainato di resnet50 su imagenet
            self.model = resnet50(weights='DEFAULT', progress=True)
            # congelo i parametri per allenare solo il layer finale
            for p in self.model.parameters():
                p.requires_grad = False
            # layer finale
            self.model.fc = nn.Linear(2048, self.outputs)

        # stampa a schermo la rete
        summary(self.model, input_size=(1, 3, 224, 224))

    def load(self):
        """Carica il modello scelto."""
        fname = next(self.ckpt_dir.glob('*.pth'))  # nome del file da caricare
        model_state = torch.load(fname, map_location=self.device)
        self.model.load_state_dict(model_state["model"])
        self.model.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward step della rete.

        :param x:           esempio di input.
        :return logits:     output prima della softmax (ci serve per calcolare la loss).
        :return outputs:    output della rete (class probabilities).
        """

        logits = self.model(x)      # output della rete prima di applicare softmax
        outputs = F.softmax(logits, dim=1)      # class probabilities

        return logits, outputs

    def train_one_epoch(self, dataloader, epoch, optimizer, criterion, writer):
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
        progress = tqdm(dataloader, total=n_batches, leave=False)
        epoch_loss = 0.0    # inizializzo la loss
        epoch_correct = 0   # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_train))
        for i, sample in enumerate(progress):
            progress.set_description(desc=f'Batch {i}/{n_batches} TRAIN')
            images, labels = sample             # __getitem__ restituisce una tupla (image, label)
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
    def validate(self, dataloader, epoch, criterion, writer):
        """
        Validazione della rete.

        :param dataloader:  il dataloader del dataset di validation.
        :param epoch:       l'epoca attuale (serve solo per salvare le metriche nel summary writer).
        :param criterion:   per la loss.
        :param writer:      per salvare le metriche.
        :return:
        """
        self.model.eval()   # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False)
        epoch_loss = 0.0    # inizializzo la loss
        epoch_correct = 0   # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0       # counter dei casi totali (sarebbe la len(dataset_val))
        for i, sample in enumerate(progress):
            progress.set_description(desc=f'Batch {i}/{n_batches} EVAL')
            batch_cases = sample.shape[0]   # numero di sample nel batch
            tot_cases += batch_cases        # accumulo il numero totale di sample
            images, labels = sample         # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            # outputs della rete
            logits, outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # loss del batch
            batch_loss = criterion(logits, labels)

            # accumulo le metriche di interesse
            epoch_loss += batch_loss.item()    # avendo usato reduction='sum' nella loss qui sto sommando la loss totale
            batch_correct = torch.sum(batch_decisions == labels)  # risposte corrette per il batch attuale
            epoch_correct += batch_correct.item()      # totale risposte corrette sull'epoca

        epoch_mean_loss = epoch_loss / tot_cases        # loss media sull'epoca
        epoch_accuracy = (epoch_correct / tot_cases) * 100.0        # accuracy sull'epoca (%)
        writer.add_scalar(f'Loss/Val', epoch_mean_loss, epoch + 1)
        writer.add_scalar(f'Accuracy/Val', epoch_accuracy, epoch + 1)

        return epoch_accuracy

    @torch.no_grad()
    def test(self, dataloader):
        """
        Valuta l'accuratezza della rete sul dataset di test.

        :param dataloader:  il dataloader del dataset di test.
        :return:
        """
        self.model.eval()  # passa in modalità eval

        n_batches = len(dataloader)
        progress = tqdm(dataloader, total=n_batches, leave=False)
        correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy
        tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_test))
        for i, sample in enumerate(progress):
            progress.set_description(desc=f'Batch {i}/{n_batches} TEST')
            batch_cases = sample.shape[0]  # numero di sample nel batch
            tot_cases += batch_cases  # accumulo il numero totale di sample
            images, labels = sample  # __getitem__ restituisce una tupla (image, label)
            images, labels = images.to(self.device), labels.to(self.device)

            # outputs della rete
            _, outputs = self.forward(images)
            # il risultato di softmax viene interpretato con politica winner takes all
            batch_decisions = torch.argmax(outputs, dim=1)

            # conto le risposte corrette
            correct += torch.sum(batch_decisions == labels)  # totale risposte corrette

        accuracy = (correct / tot_cases) * 100.0  # accuracy sull'epoca (%)

        return accuracy

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
        progress = tqdm(range(epochs), total=epochs, leave=False)
        # creo un summary writer per salvare le metriche (loss e accuracy)
        writer = SummaryWriter(log_dir=str(ckpt_dir))

        # inizializzo per scegliere il modello migliore
        best_acc = 0.0
        for epoch in progress:
            # train
            progress.set_description(desc=f'{epoch + 1}/{epochs}')
            # alleno la rete su tutti gli esempi del train set (1 epoca)
            self.train_one_epoch(train_loader, epoch, optimizer, criterion, writer)
            # valido il modello attuale sul validation set e ottengo l'accuratezza attuale
            acc_now = self.validate(val_loader, epoch, criterion, writer)
            # scelgo il modello migliore e lo salvo
            if acc_now > best_acc:
                best_acc = acc_now
                best_epoch = epoch + 1

                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                    'accuracy': best_acc,
                }, f=ckpt_dir / f'classifier_e{best_epoch}.pth')

        # restituiso le metriche per stamparle e fare la media sui fold
        best_metrics = {'epoch': best_epoch, 'accuracy': best_acc}

        return best_metrics


def main(args):
    ROOT = Path(args.root)
    MODE = args.mode
    N_FOLDS = args.n_folds
    DEVICE = args.device
    BACKBONE = args.backbone
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LR = args.lr
    CHECKPOINT_DIR = Path(args.checkpoint_dir)

    assert MODE in ['train', 'eval'], '--mode deve essere uno tra "train" e "eval".'
    assert DEVICE in ['cuda', 'gpu'], '--device deve essere uno tra "cuda" e "gpu".'
    assert BACKBONE in ['cnn', 'resnet'], 'le --backbone disponibili sono: "cnn" e "resnet".'

    if BACKBONE == 'resnet':
        train_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.RandomAffine(45, translate=(0.1, 0.1),
                                                                       scale=(0.8, 1.2), fill=255),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                                                    std=torch.tensor([0.229, 0.224, 0.225]))])
        val_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                                                  std=torch.tensor([0.229, 0.224, 0.225]))])
    elif BACKBONE == 'cnn':
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
            val_ds.set_transforms(val_transforms)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # gli altri split li uso per il train, mettendoli in una lista e passandoli a Concat, settando prima le
            # train_transform
            train_datasets = [j for j in splits if j is not split]  # come train uso gli altri, unendoli con Concat
            for d in train_datasets:
                d.set_transforms(train_transforms)
            train_ds = Concat(train_datasets)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

            cls = Classifier(BACKBONE, DEVICE, actual_dir)      # inizializzo il classificatore
            train_result = cls.train(train_loader, val_loader, i, EPOCHS, LR)      # alleno
            best_results.append(train_result)

        accuracies = [r["accuracy"] for r in best_results]  # elenco le best_accuracy di ogni fold per farne la media
        mean_accuracy = np.mean(accuracies)
        for i, r in enumerate(best_results):
            print(f'Fold {i+1}: miglior accuratezza raggiunta dopo {r["epoch"]} epoche pari al {r["accuracy"]}%.')
        print(f'Accuracy media: {mean_accuracy}')

    else:
        # a questo giro deve essere il percorso completo alla cartella in cui sono stati salvati i progressi
        # del modello prescelto
        actual_dir = CHECKPOINT_DIR
        # per creare il dataset non passo il parametro split perchè non serve (__init__ lo setta a n_folds)
        test_ds = MyDataset(ROOT, N_FOLDS, mode=MODE, transforms=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        cls = Classifier(BACKBONE, DEVICE, actual_dir)  # inizializzo il classificatore
        cls.load()

        test_accuracy = cls.test(test_loader)
        print(f'Accuracy sui dati di test: {test_accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train del classificatore',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\subset',
                        help='Root del dataset.')
    parser.add_argument('--n-folds', type=int, default=3, help='Numero di fold per la cross validation')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Scegliere tra "train" e "eval" in base a quale modalità si desidera.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Scegliere se usare la gpu ("cuda") o la "cpu".')
    parser.add_argument('-b', '--backbone', type=str, default='resnet',
                        help='Scegliere se utilizzare una semplice "cnn" o la "resnet" come features extractor.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Epoche per eseguire il train')
    parser.add_argument('--batch-size', type=int, default=16, help='Numero di esempi in ogni batch.')
    parser.add_argument('--num-workers', type=int, default=3, help='Numero di worker.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--checkpoint_dir', type=str, default='runs/classifier',
                        help='Cartella dove salvare i risultati dei vari esperimenti. Se --mode == "train" specificare'
                             ' la cartella madre che contiene tutte le annotazioni sugli esperimenti; se --mode =='
                             ' "eval" indicare la cartella dello specifico esperimento che si vuole valutare.')

    arguments = parser.parse_args()
    main(arguments)
