import random

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAffine, ToTensor, Normalize
from torchinfo import summary

from easyfsl.datasets import FewShotDataset
from easyfsl.methods import FewShotClassifier
from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks, MatchingNetworks, RelationNetworks
from easyfsl.modules import resnet18, resnet10, resnet12, resnet50


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
        random.shuffle(self.classes)
        self.classes_in_split = self.classes[start: end]
        ids_in_split = self.annos.index[self.annos['item_id'].isin(self.classes_in_split)].tolist()

        return ids_in_split

    def get_labels(self):
        return self.labels

    def number_of_classes(self):
        return len(self.classes_in_split)


class Concat(FewShotDataset):
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
        self.labels = []
        self.classes_in_split = []
        for d in self.datasets:
            self.labels.extend(d.labels)
            self.classes_in_split.extend(d.classes_in_split)
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

    def get_labels(self):
        return self.labels

    def number_of_classes(self):
        return len(self.classes_in_split)


def train_one_epoch(model: FewShotClassifier, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, method: str, n_way: int):
    """
    Allena la rete per un'epoca.

    :param model:       il modello(già sul giusto device).
    :param dataloader:  il dataloader del dataset di train.
    :param optimizer:   per aggiornare i parametri.
    :param criterion:   per la loss.
    :param device:      cuda o cpu.
    :param method:      metodo usato.
    :param n_way:       numero di classi.
    :return:
    """
    model.train()  # modalità train
    optimizer.zero_grad()  # svuoto i gradienti
    n_episodes = len(dataloader)
    progress = tqdm(dataloader, total=n_episodes, leave=False, desc='COMPLETED EPISODES')
    epoch_loss = 0.0  # inizializzo la loss
    tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_val))
    for sample in progress:
        support_images, support_labels, query_images, query_labels, _ = sample
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        episode_cases = query_labels.shape[0]
        tot_cases += episode_cases  # accumulo il numero totale di sample
        # output della rete
        model.process_support_set(support_images, support_labels)
        query_images = query_images.to(device)
        classification_scores = model(query_images)
        if method == 'rel':
            query_labels = F.one_hot(query_labels, num_classes=n_way).type(torch.float)
        # loss del batch e backward step
        query_labels = query_labels.to(device)
        episode_loss = criterion(classification_scores, query_labels)
        episode_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # accumulo le metriche di interesse
        epoch_loss += episode_loss.item()  # avendo usato reduction='sum' nella loss qui sto sommando la loss totale
        postfix = {'batch_mean_loss': episode_loss.item() / query_labels.shape[0]}
        progress.set_postfix(postfix)

    epoch_mean_loss = epoch_loss / tot_cases        # loss media sull'epoca
    return epoch_mean_loss


@torch.no_grad()
def validate(model: FewShotClassifier, val_loader: DataLoader, device: torch.device):
    model.eval()  # passa in modalità eval

    n_episodes = len(val_loader)
    progress = tqdm(val_loader, total=n_episodes, leave=False, desc='EVAL')
    epoch_correct = 0  # segno le prediction corrette della rete per poi calcolare l'accuracy
    tot_cases = 0  # counter dei casi totali (sarebbe la len(dataset_val))
    for sample in progress:
        support_images, support_labels, query_images, query_labels, _ = sample
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)

        episode_cases = query_labels.shape[0]  # numero di sample nel batch
        tot_cases += episode_cases  # accumulo il numero totale di sample
        # output della rete
        model.process_support_set(support_images, support_labels)
        episode_predictions = model(query_images).detach().data
        correct_predictions = torch.sum(torch.argmax(episode_predictions, dim=1) == query_labels)

        epoch_correct += correct_predictions.item()
        postfix = {'batch_accuracy': (correct_predictions.item() / episode_cases) * 100.0}
        progress.set_postfix(postfix)

    epoch_accuracy = (epoch_correct / tot_cases) * 100.0

    return epoch_accuracy


def train(epochs: int, model: FewShotClassifier, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, criterion: nn.Module,
          device: torch.device, ckpt_dir: Path, method: str, n_way: int):
    """

    :param epochs:          numero di epoche.
    :param model:           modello per fare il few shot learning.
    :param train_loader:    dataloader del dataset di train (serve solo a train__one_epoch).
    :param val_loader:      dataloader del dataset di validation (serve solo a evaluate).
    :param optimizer:       per aggiornare i parametri (serve solo a train__one_epoch).
    :param scheduler:       per variare il learning rate durante il training.
    :param criterion:       par calcolare la loss (serve solo a train__one_epoch).
    :param device:          cuda o cpu.
    :param ckpt_dir:        la directory dove stiamo salvando il modello.
    :param method:          metodo usato.
    :param n_way:           numero di classi.
    :return:
    """
    writer = SummaryWriter(log_dir=str(ckpt_dir))
    best_acc = 0.0
    progress = tqdm(range(epochs), total=epochs, leave=False, desc='COMPLETED EPOCHS')
    for epoch in progress:
        # alleno la rete su tutti gli esempi del train set (1 epoca)
        epoch_mean_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, method, n_way)
        writer.add_scalar(f'Loss/Train', epoch_mean_loss, epoch + 1)
        # valido il modello attuale sul validation set e ottengo l'accuratezza attuale
        acc_now = validate(model, val_loader, device)
        writer.add_scalar(f'Accuracy/Val', acc_now, epoch + 1)
        if scheduler:
            scheduler.step()
        # scelgo il modello migliore e lo salvo
        if acc_now > best_acc:
            best_acc = acc_now
            best_epoch = epoch + 1

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': best_epoch,
                'accuracy': best_acc,
            }, f=ckpt_dir / 'classifier.pth')
    # restituiso le metriche per stamparle e fare la media sui fold
    best_metrics = {'epoch': best_epoch, 'accuracy': best_acc}

    return best_metrics


def main(args):
    ROOT = Path(args.root)
    N_FOLDS = args.n_folds
    MODE = args.mode
    DEVICE = args.device
    BACKBONE = args.backbone
    EPOCHS = args.epochs
    EPISODES = args.episodes
    VAL_EPISODES = args.val_episodes
    TEST_EPISODES = args.test_episodes
    N_WAY = args.n_way
    K_SHOT = args.k_shot
    N_QUERY = args.n_query
    NUM_WORKERS = args.num_workers
    LR = args.lr
    SEED = args.seed
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    NOHUP = args.nohup
    METHOD = args.method
    WEIGHTS = Path(args.weights)

    assert MODE in ['train', 'eval'], '--mode deve essere uno tra "train" e "eval".'
    assert DEVICE in ['cuda', 'cpu'], '--device deve essere uno tra "cuda" e "cpu".'
    assert BACKBONE in ['resnet', 'resnet18', 'resnet10', 'resnet12'],\
        'le --backbone disponibili sono: "resnet" (50), "resnet18", "resnet10" e "resnet12".'
    assert METHOD in ['proto', 'match', 'rel'], 'i metodi di few-shot learning utilizzabili sono prototypical ' \
                                                'network ("proto"), matching network ("match") o relation network ' \
                                                '("rel").'

    backbones = {'resnet': {'arch': resnet50(), 'f_dim': 2048},
                 'resnet18': {'arch': resnet18(), 'f_dim': 512},
                 'resnet10': {'arch': resnet10(), 'f_dim': 512},
                 'resnet12': {'arch': resnet12(), 'f_dim': 640}}
    bb = backbones[BACKBONE]['arch']
    f_dim = backbones[BACKBONE]['f_dim']

    transforms = Compose([ToTensor(),
                          RandomAffine(45, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=255),
                          RandomHorizontalFlip(p=0.5),
                          Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                    std=torch.tensor([0.229, 0.224, 0.225]))])

    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    #
    # print(f'total: {t}\nreserved: {r}\nallocated: {a}')

    if MODE == 'train':
        if not NOHUP:
            # creo una cartella dove salverò l'andamento dei vari allenamenti, serve solo se sto trainando
            CHECKPOINT_DIR = Path('runs/fsl')
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
        # TRAIN E VALIDAZIONE
        splits = []     # qui salvo gli n_folds dataset che sono i singoli split
        best_results = []
        for i in tqdm(range(N_FOLDS), total=N_FOLDS, desc='Creo gli split del dataset.'):
            splits.append(MyDataset(ROOT, N_FOLDS, i, mode=MODE, transforms=transforms, seed=SEED))

        for i, split in tqdm(enumerate(splits), total=N_FOLDS, desc='COMPLETED FOLDS'):
            # ciclicamente uso uno split come val
            val_ds = split  # split è il dataset che sto usando come validation

            # gli altri split li uso per il train, mettendoli in una lista e passandoli a Concat# train_transform
            train_datasets = [j for j in splits if j is not split]  # come train uso gli altri, unendoli con Concat
            train_ds = Concat(train_datasets)

            # sampler che genera i vari episodi per il train e la validazione
            train_sampler = TaskSampler(train_ds, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=EPISODES)
            val_sampler = TaskSampler(val_ds, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=VAL_EPISODES)

            # collate_fn custom che restituisce (support_images, support_labels, query_images, query_labels, class_ids)
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                      collate_fn=train_sampler.episodic_collate_fn)
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True,
                                    collate_fn=val_sampler.episodic_collate_fn)

            # adesso distinguo il metodo da usare
            if METHOD == 'proto':
                # la prototypical è la più normale di tutte
                classifier = PrototypicalNetworks(bb).to(DEVICE)
                criterion = nn.CrossEntropyLoss(reduction='sum')
            elif METHOD == 'rel':
                # relation network vuole la featuremap prima del flatten, quindi bisogna dire alla backbone di non fare
                # il pooling e impostare come feature_dimension la prima dimensione della feature map che è la depth
                bb.use_pooling = False
                classifier = RelationNetworks(backbone=bb, feature_dimension=f_dim).to(DEVICE)
                criterion = nn.MSELoss(reduction='sum')
            elif METHOD == 'match':
                classifier = MatchingNetworks(backbone=bb, feature_dimension=f_dim).to(DEVICE)
                criterion = nn.NLLLoss(reduction='sum')

            # adesso ho il classificatore e la loss function, mi manca da definire l'optimizer e il summary per il log
            # dei dati del train
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), LR,
                                         eps=1e-6, weight_decay=5e-4)
            scheduler = None

            # optimizer = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)
            ckpt_dir = actual_dir / f'fold_{i}'

            best_metrics = train(EPOCHS, classifier, train_loader, val_loader, optimizer, scheduler, criterion,
                                 DEVICE, ckpt_dir, METHOD, N_WAY)
            best_results.append(best_metrics)

        for i, r in enumerate(best_results):
            print(f'Fold {i + 1}: miglior accuratezza raggiunta dopo {r["epoch"]} epoche pari al {r["accuracy"]}%.')
        accuracies = [r["accuracy"] for r in best_results]  # elenco le best_accuracy di ogni fold per la media
        mean_accuracy = np.mean(accuracies)
        print(f'Accuracy media: {mean_accuracy}%.')

    else:
        # a questo giro deve essere il percorso alla cartella dell'esperimento
        experiment_dir = CHECKPOINT_DIR
        # per creare il dataset passo il parametro split ma non serve (__init__ lo setta a n_folds)
        test_ds = MyDataset(ROOT, N_FOLDS, split=0, mode=MODE, transforms=transforms, seed=SEED)
        test_sampler = TaskSampler(test_ds, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=TEST_EPISODES)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler, num_workers=NUM_WORKERS, pin_memory=False,
                                 collate_fn=test_sampler.episodic_collate_fn)
        # adesso distinguo il metodo da usare
        if METHOD == 'proto':
            # la prototypical è la più normale di tutte
            classifier = PrototypicalNetworks(bb).to(DEVICE)
        elif METHOD == 'rel':
            # relation network vuole la featuremap prima del flatten, quindi bisogna dire alla backbone di non fare
            # il pooling e impostare come feature_dimension la prima dimensione della feature map che è la depth
            bb.use_pooling = False
            classifier = RelationNetworks(backbone=bb, feature_dimension=f_dim).to(DEVICE)
        elif METHOD == 'match':
            classifier = MatchingNetworks(backbone=bb, feature_dimension=f_dim).to(DEVICE)

        test_acc = []
        for i in range(N_FOLDS):
            weights = experiment_dir / f'fold_{i}' / 'classifier.pth'
            model_state = torch.load(weights, map_location=DEVICE)
            classifier.load_state_dict(model_state["model"])
            fold_acc = validate(classifier, test_loader, DEVICE, METHOD, N_WAY)
            test_acc.append(fold_acc)
            print(f'Accuracy sui dati di test durante il fold {i + 1}: {fold_acc}%.')
        print(f'Accuracy media: {torch.mean(torch.tensor(test_acc))}%.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-Shot learning con episodic training.',
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
                        help='Per decidere se allenare anche la resnet o semplicemente caricare i pesi. (I)')
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
