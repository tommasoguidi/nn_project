import pandas as pd
import os
import gzip
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import cv2


def drop_by_ratio(imgs_metadata: pd.DataFrame, ratio: int):
    """
    Prende il dataframe corretto con i metadati delle singole immagini e rimuove tutte quelle che non rispettano il
    vincolo sull'aspect ratio.

    :param imgs_metadata:   dataframe GIA' CORRETTO con dimensioni e path immagini.
    :param ratio:           tutte le immagini che hanno un rapporto max_dim/min_dim inferiore a ratio vengono scartate.
    :return imgs_metadata:  dataframe filtrato dalle immagini secondo la loro aspect ratio.
    """
    dropped_imgs = []
    tot_imgs = imgs_metadata.shape[0]
    # ciclo sulle righe
    for row in tqdm(imgs_metadata.iterrows(), total=tot_imgs, desc="Scarto le immagini secondo l'aspect ratio."):
        index = row[0]  # indice della riga
        data = row[1]   # annotazioni

        h = data['height']
        w = data['width']
        max_dim = max(h, w)
        min_dim = min(h, w)
        if max_dim / min_dim > ratio:
            dropped_imgs.append(index)  # aggiungo l'indice della riga alla lista di quelli da scartare

    imgs_metadata.drop(index=dropped_imgs, inplace=True)    # scarto le righe con indice in dropped_images

    return imgs_metadata


def get_useful_listings(listings_dir: Path,
                        dest_dir: Path,
                        imgs_metadata: pd.DataFrame,
                        min_images: int,
                        n_cat: int):
    """
    La funzione ritorna un dataframe i cui index sono i singoli image_id di ciascun prodotto che è stato selezionato
    nel subset (almeno min_images immagini dopo aver rimosso quelle che non rispettano il vincolo su --ratio,
    appartenenti alle top n_cat). Dopo aver selezionato le n_cat classi, per bilanciare un po' il dataset, che risulta
    fortemente sbilanciato, viene eseguito un downsampling delle classi con un numero di prodotti superiore alla media.

    :param listings_dir:        directory dei metadati delle inserzioni.
    :param dest_dir:            directory di destinazione del subset.
    :param imgs_metadata:       il dataframe contenente le annotazioni corrette sulle immagini.
    :param min_images:          le inserzioni con un numero di immagini inferiore a min_images vengono scartate.
    :param n_cat:               per selezionare solo le n_cat classi più rappresentate.
    :return useful_listings:    dataframe contenente le inserzioni rilevanti
    """
    # apro tutti i file delle annotazioni, che sono divisi in più archivi, e li unisco in un unico dataframe
    listings = os.listdir(listings_dir)
    useful_listings = pd.DataFrame()
    n_listings = len(listings)
    useful_index = set(imgs_metadata.index.tolist())
    for i in tqdm(listings, total=n_listings, desc='Estraggo le informazioni delle inserzioni'):
        archive = listings_dir / i
        with gzip.open(archive, 'rb') as f:
            data = pd.read_json(f, lines=True)
        # tengo solo le colonne che mi interessano
        data = data[['item_id', 'product_type', 'main_image_id', 'other_image_id']]
        # estraggo i dati che mi servono in un formato più leggibile
        data['product_type'] = data['product_type'].apply(lambda x: x[0]['value'])
        # sostituisco i valori nan con liste vuote
        data['other_image_id'] = data['other_image_id'].apply(lambda x: [] if x is np.nan
                                                              else [i for i in x if i in useful_index])
        # per mettere tutti gli image_id associati a un prodotto nella stessa lista, di default l'immagine principale
        # è salvata nella colonna 'main_image_id' mentre le altre sono in una lista sotto la colonna 'other_image_id'
        data.apply(lambda x: x['other_image_id'].append(x['main_image_id']) if x['main_image_id'] in useful_index
                   else x['other_image_id'], axis=1)
        data.rename(columns={'other_image_id': 'image_id'}, inplace=True)   # scarto la colonna perchè non serve più
        data.drop(columns='main_image_id', inplace=True)
        data['image_id'] = data['image_id'].apply(lambda x: tuple(x))
        useful_listings = pd.concat([useful_listings, data], ignore_index=True)     # metto tutto in unico dataframe

    # alcune inserzioni hanno i metadati in più lingue, quindi ci sono delle righe duplicate ma questi a noi non
    # interessano quindi li scartiamo
    useful_listings.drop_duplicates(inplace=True, ignore_index=True)
    # dato che alcune inserzioni usano immagini stock comuni, rimuovo quelle che si ripetono
    useful_listings = useful_listings.explode('image_id')
    useful_listings.drop_duplicates(subset='image_id', keep=False, inplace=True, ignore_index=True)
    # tolgo le inserzioni con meno di min_images immagini, per farlo raggruppo nuovamente per 'item_id' e 'product_type'
    useful_listings = useful_listings.groupby(['item_id', 'product_type']).agg({'image_id': lambda x: x.tolist()})
    useful_listings.reset_index(inplace=True)
    # dopo aver rimesso tutte le immagini di un prodotto in una lista, scarto quelli con meno di min_images
    useful_listings = useful_listings[useful_listings['image_id'].str.len() >= min_images]
    useful_listings.reset_index(drop=True, inplace=True)

    # adesso scrivo i metadati rimasti nella loro forma finale, usando come index la 'image_id' e integrando le
    # informazioni presenti nel csv dei metadati delle immagini
    useful_listings = useful_listings.explode('image_id').set_index('image_id')
    useful_listings = useful_listings.merge(imgs_metadata, how='left', on='image_id')

    # adesso vanno selezionate le immagini delle n_cat classi più rappresentate
    # value_counts restituisce una serie con il numero di volte in cui un'entrata compare nella colonna 'product_type'
    # in ordine decrescente, le classi selezionate saranno dunque gli indici di questa serie
    top_n = useful_listings['product_type'].value_counts()[0: n_cat]
    top_n_classes = top_n.index.tolist()
    top_n_imgs_per_class = top_n.tolist()
    # tengo solo le inserzioni il cui 'product_type' è tra i top_n
    useful_listings = useful_listings[useful_listings['product_type'].isin(top_n_classes)]

    # essendo molto sbilanciato, andremo a fare un subsampling delle classi più rappresentate, per evitare di tenere
    # nel dataset prodotti con meno di min_images immagini, faremo il sampling sui prodotti invece che sulle immagini
    # quello che segue è un approccio molto conservativo, potrebbe essere combinato ad un upsampling delle classi
    # meno rappresentate
    avg = int(np.mean(top_n_imgs_per_class))     # numero medio di immagini in ciascuna delle top_n classi
    product_kept = []   # lista degli item_id da tenere nel subset
    for _class, _n_imgs in zip(top_n_classes, top_n_imgs_per_class):
        # filtro il dataframe per tenere solo le righe della specifica classe
        _class_listings = useful_listings[useful_listings['product_type'] == _class]
        # salvo in una lista gli item_id unici, se la classe è sottorappresentata li tengo tutti, altrimenti sampling
        _products = _class_listings['item_id'].unique().tolist()
        if _n_imgs > avg:
            _p = avg / _n_imgs   # probabilità di tenere il prodotto
            _u = np.random.uniform(size=_n_imgs)     # campione di distribuzione uniforme
            _choice = _u < _p  # array di booleani
            # tengo solo i prodotti per cui l'elemento di _choice con stesso indice è True
            _products = [p for p, b in zip(_products, _choice) if b]
        # una volta filtrati aggiungo i prodotti alla lista dei prodotti da tenere
        product_kept.extend(_products)

    # finalmente tutte e sole le inserzioni ceh ci interessano
    final_metadata = useful_listings[useful_listings['item_id'].isin(product_kept)]
    f_name = dest_dir / 'subset_10_images.csv'
    final_metadata.to_csv(f_name)   # salvo le annotazioni
    return final_metadata


def save_images(imgs_dir: Path, dest_dir: Path, metadata: pd.DataFrame, size: int):
    """
    Rendo quadrate le immagini che non lo sono aggiungendo pixel bianchi, poi ridimensiono in modo che la dimensione
    di ciascun lato sia pari a size. Infine salvo le immagini nella cartella desiderata.

    :param imgs_dir:    cartella contenente le immagini originali.
    :param dest_dir:    destinazione in cui salvare le immagini preprocessate.
    :param metadata:    dataframe già modificato, contenente tutte e sole le immagini da tenere nel subset.
    :param size:        dimensione desiderata delle immagini (quadrate).
    :return:
    """
    index = metadata.index.tolist()
    tot_index = len(index)
    dest_dir = dest_dir / 'images'
    for idx in tqdm(index, total=tot_index, desc='Salvataggio delle immagini nel formato desiderato'):
        data = metadata.loc[idx]    # prendo la riga corrispondente a idx
        h = data['height']
        w = data['width']
        path = imgs_dir / data['path']
        img = cv2.imread(str(path))
        # per fare il padding seleziono la dimensione massima e aggiungo pixel bianchi lungo l'altra
        new_dim = max(h, w)
        if h != w:  # in questo caso l'immagine è già quadrata
            # come background userò un'immagine (in realtà un array numpy) di pixel bianchi
            bg = np.ones((new_dim, new_dim, 3), dtype=np.uint8) * 255
            # calcolo le coordinate per centrare la vecchia immagine sullo sfondo bianco
            x_semi_offset = (new_dim - w)//2
            y_semi_offset = (new_dim - h) // 2
            # le sovrappongo
            bg[y_semi_offset:y_semi_offset+h, x_semi_offset:x_semi_offset+w, :] = img
            img = bg

        # resize (tranne il caso in cui sia già della giusta dimensione), serve a poco, ma opencv consiglia di usare due
        # metodi diversi se si fa downscaling o upscaling
        if new_dim > size:
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        elif new_dim < size:
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

        # salviamo l'immagine
        dest_fname = dest_dir / data['path']
        # 'path' è nella forma subdir/image_id.jpg, quindi estraggo subdir con split()
        dest_subdir = dest_dir / (data['path'].split('/')[0])
        os.makedirs(dest_subdir, exist_ok=True)     # crea la cartella e anche quelle intermedie se non esistono ancora
        cv2.imwrite(str(dest_fname), resized)


def main(args):
    # definizione dei percorsi
    pd.set_option('display.max_columns', None)
    ROOT = Path(args.root)  # root del dataset
    DEST = Path(args.dest)  # cartella di destinazione del subset
    RATIO = args.ratio  # aspect ratio secondo cui scartare le immagini (spiego meglio in drop_by_ratio())
    SIZE = args.size    # dimensione a cui eseguire il resize dell'immagine
    MIN_IMAGES = args.min_images    # numero di immagini minimo per tenere un prodotto
    N_CAT = args.n_cat  # numero di categorie merceologiche da tenere
    listings_dir = ROOT / 'abo-listings/listings/metadata'  # cartella delle annotazioni
    imgs_dir = ROOT / 'abo-images-small/images/small'   # cartella in cui salvare le immagini modificate

    # leggo il csv corretto usando fix_csv.py
    imgs_metadata = pd.read_csv(ROOT / 'abo-images-small/images/metadata/fixed_images.csv', index_col='image_id')
    # filtro le immagini secondo l'aspect ratio desiderata
    imgs_metadata = drop_by_ratio(imgs_metadata, RATIO)
    # print(imgs_metadata)
    # vengono filtrate le immagini e creato il file di annotazioni del subset
    metadata = get_useful_listings(listings_dir, DEST, imgs_metadata, MIN_IMAGES, N_CAT)
    # print(metadata)
    # metadata = pd.read_csv(dest_dir / 'subset_images.csv', index_col='image_id')
    # salva le immagini rimaste nel file csv del subset
    save_images(imgs_dir, DEST, metadata, SIZE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\ABO_DATASET',
                        help='Root del dataset.')
    parser.add_argument('--dest', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\subset_10',
                        help='Directory in cui salvare il dataset preprocessato.')
    parser.add_argument('--ratio', type=int, default=5, help='Soglia per scartare le immagini con rapporto '
                                                             'max_dim/min_dim superiore a quello indicato.')
    parser.add_argument('--size', type=int, default=224, help='Dimensione finale delle immagini (quadrate).')
    parser.add_argument('--min-images', type=int, default=5, help='Scarta le inserzioni con un numero di immagini '
                                                                  'inferiore per poter fare crossvalidation.')
    parser.add_argument('--n-cat', type=int, default=10, help='Per selezionare solo le classi più rappresentate.')
    arguments = parser.parse_args()
    main(arguments)
