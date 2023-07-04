import pandas as pd
import os
import gzip
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import cv2


def drop_by_ratio(imgs_metadata: pd.DataFrame, args: argparse.ArgumentParser):
    """
    Prende il dataframe corretto con i metadati delle singole immagini e rimuove tutte quelle che non rispettano il
    vincolo sull'aspect ratio

    :param imgs_metadata:   dataframe GIA' CORRETTO con dimensioni e path immagini
    :param args:            argomenti passati da command line
    :return:                dataframe filtrato dalle immagini secondo la loro aspect ratio
    """
    dropped_imgs = []
    tot_imgs = imgs_metadata.shape[0]
    for row in tqdm(imgs_metadata.iterrows(), total=tot_imgs, desc="Scarto le immagini secondo l'aspect ratio"):
        index = row[0]
        data = row[1]

        h = data['height']
        w = data['width']
        max_dim = max(h, w)
        min_dim = min(h, w)
        if max_dim / min_dim > args.ratio:
            dropped_imgs.append(index)

    imgs_metadata.drop(index=dropped_imgs, inplace=True)

    return imgs_metadata


def get_useful_listings(listings_dir: Path, dest_dir: Path, imgs_metadata: pd.DataFrame, args: argparse.ArgumentParser):
    """
    La funzione ritorna un dataframe con index i singoli image_id di ciascun prodotto che è stato selezionato
    nel subset (almeno --folds immagini dopo aver rimosso quelle che non rispettano il vincolo su --ratio, appartenenti
    alle top --n-cat)

    :param listings_dir:        directory dei metadati delle inserzioni
    :param dest_dir:            directory di destinazione del subset
    :param imgs_metadata:       il dataframe contenente le annotazioni corrette sulle immagini
    :param args:                argomenti passati da command line
    :return useful_listings:    dataframe contenente le inserzioni rilevanti
    """
    listings = os.listdir(listings_dir)
    useful_listings = pd.DataFrame()
    n_listings = len(listings)
    useful_index = set(imgs_metadata.index.tolist())
    for i in tqdm(listings, total=n_listings, desc='Estraggo le infromazioni delle inserzioni'):
        archive = listings_dir / i
        with gzip.open(archive, 'rb') as f:
            data = pd.read_json(f, lines=True)
        # tengo solo le colonne che mi interessano
        data = data[['item_id', 'product_type', 'main_image_id', 'other_image_id']]
        # estraggo i dati che mi servono in un formato più leggibile
        data['product_type'] = data['product_type'].apply(lambda x: x[0]['value'])
        data['other_image_id'] = data['other_image_id'].apply(lambda x: [] if x is np.nan
                                                              else [i for i in x if i in useful_index])
        # per mettere tutti gli image_id associati a un prodotto nella stessa lista
        data.apply(lambda x: x['other_image_id'].append(x['main_image_id']) if x['main_image_id'] in useful_index
                   else x['other_image_id'], axis=1)
        data.rename(columns={'other_image_id': 'image_id'}, inplace=True)
        data.drop(columns='main_image_id', inplace=True)
        data['image_id'] = data['image_id'].apply(lambda x: tuple(x))
        useful_listings = pd.concat([useful_listings, data], ignore_index=True)

    # alcune inserzioni hanno i metadati in più lingue, quindi ci sono delle righe duplicate ma questi a noi non
    # interessano quindi li scartiamo
    useful_listings.drop_duplicates(inplace=True, ignore_index=True)
    # dato che alcune inserzioni usano immagini stock comuni, rimuovo quelle che si ripetono
    useful_listings = useful_listings.explode('image_id')
    useful_listings.drop_duplicates(subset='image_id', keep=False, inplace=True, ignore_index=True)
    # tolgo le inserzioni con meno di --folds immagini, per farlo raggruppo nuovamente per 'item_id' e 'product_type'
    useful_listings = useful_listings.groupby(['item_id', 'product_type']).agg({'image_id': lambda x: x.to_list()})
    useful_listings.reset_index(inplace=True)
    useful_listings = useful_listings[useful_listings['image_id'].str.len() >= args.min_images]
    useful_listings.reset_index(drop=True, inplace=True)

    # adesso scrivo i metadati rimasti nella loro forma finale, usando come index la 'image_id' e integrando le
    # informazioni presenti nel csv dei metadati delle immagini
    useful_listings = useful_listings.explode('image_id').set_index('image_id')
    useful_listings = useful_listings.merge(imgs_metadata, how='left', on='image_id')

    # adesso vanno selezionate le immagini delle --n-cat classi più rappresentate
    # value_counts restituisce una serie con il numero di volte in cui un'entrata compare nella colonna 'product_type'
    # in ordine decrescente, le classi selezionate saranno dunque gli indici di questa serie
    top_n = useful_listings['product_type'].value_counts()[0: args.n_cat]
    top_n = top_n.index.to_list()

    # finalmente tutte e sole le inserzioni ceh ci interessano
    final_metadata = useful_listings[useful_listings['product_type'].isin(top_n)]
    f_name = dest_dir / 'subset_images.csv'
    final_metadata.to_csv(f_name)
    return final_metadata


def save_images(imgs_dir: Path, dest_dir: Path, metadata: pd.DataFrame, args: argparse.ArgumentParser):
    """

    :param imgs_dir:
    :param dest_dir:
    :param metadata:
    :param args:
    :return:
    """
    index = metadata.index.tolist()
    tot_index = len(index)
    dest_dir = dest_dir / 'images'
    for idx in tqdm(index, total=tot_index, desc='Salvataggio delle immagini nel formato desiderato'):
        data = metadata.loc[idx]
        h = data['height']
        w = data['width']
        path = imgs_dir / data['path']
        img = cv2.imread(str(path))
        new_dim = max(h, w)
        if h != w:
            bg = np.ones((new_dim, new_dim, 3), dtype=np.uint8)*255

            x_semi_offset = (new_dim - w)//2
            y_semi_offset = (new_dim - h) // 2

            bg[y_semi_offset:y_semi_offset+h, x_semi_offset:x_semi_offset+w, :] = img
            img = bg

        # resize (tranne il caso in cui sia già della giusta dimensione)
        if new_dim > args.size:
            resized = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        elif new_dim < args.size:
            resized = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

        # salviamo l'immagine
        dest_fname = dest_dir / data['path']
        dest_subdir = dest_dir / (data['path'].split('/')[0])
        if not os.path.exists(dest_subdir):
            os.mkdir(dest_subdir)
        cv2.imwrite(str(dest_fname), resized)



def main(args):
    # definizione dei percorsi
    pd.set_option('display.max_columns', None)
    root = Path(args.root)
    dest_dir = Path(args.dest)
    listings_dir = root / 'abo-listings/listings/metadata'
    imgs_dir = root / 'abo-images-small/images/small'

    imgs_metadata = pd.read_csv(root / 'abo-images-small/images/metadata/fixed_images.csv', index_col='image_id')
    # filtro le immagini secondo l'aspect ratio desiderata
    imgs_metadata = drop_by_ratio(imgs_metadata, args)
    # print(imgs_metadata)

    # vengono filtrate le immagini e creato il file di annotazioni del subset
    metadata = get_useful_listings(listings_dir, dest_dir, imgs_metadata, args)
    # print(metadata)

    # metadata = pd.read_csv(dest_dir / 'subset_images.csv', index_col='image_id')
    save_images(imgs_dir, dest_dir, metadata, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ratio', type=int, default=5, help='Soglia per scartare le immagini con rapporto '
                                                             'max_dim/min_dim superiore a quello indicato.')
    parser.add_argument('--size', type=int, default=224, help='Dimensione finale delle immagini (quadrate).')
    parser.add_argument('--min-images', type=int, default=5, help='Scarta le inserzioni con un numero di immagini '
                                                                  'inferiore per poter fare crossvalidation.')
    parser.add_argument('--n-cat', type=int, default=50, help='Per selezionare solo le classi più rappresentate.')
    parser.add_argument('--root', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\ABO_DATASET',
                        help='Root del dataset.')
    parser.add_argument('--dest', type=str, default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\subset',
                        help='Directory in cui salvare il dataset preprocessato.')
    arguments = parser.parse_args()
    main(arguments)
