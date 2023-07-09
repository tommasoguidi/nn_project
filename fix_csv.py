import pandas as pd
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path


def main(args):
    file_dir = Path(args.file_dir)  # percorso del file .csv
    imgs_dir = Path(args.imgs_dir)  # percorso della cartella con le immagini

    imgs_metadata = pd.read_csv(file_dir / 'images.csv', index_col='image_id')  # leggo il csv

    tot_imgs = imgs_metadata.shape[0]
    # ciclo su ogni riga del dataframe
    for row in tqdm(imgs_metadata.iterrows(), total=tot_imgs, desc="Scarto le immagini secondo l'aspect ratio"):
        index = row[0]  # indice della riga
        data = row[1]   # annotazioni nelle colonne

        img_path = imgs_dir / data['path']
        img = cv2.imread(str(img_path))     # apro l'immagine
        h, w, _ = img.shape
        # sostituisco nel csv i valori corretti di altezza e larghezza
        imgs_metadata.loc[index, 'width'] = w
        imgs_metadata.loc[index, 'height'] = h

    f_name = file_dir / 'fixed_images.csv'
    imgs_metadata.to_csv(f_name)    # salvo il csv corretto


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file-dir', type=str,
                        default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\ABO_DATASET\abo-images-small\images\metadata',
                        help='Directory del csv.')
    parser.add_argument('--imgs-dir', type=str,
                        default=r'C:\Users\rx571gt-b034t\Desktop\PROJECT\ABO_DATASET\abo-images-small\images\small',
                        help='Directory del csv.')
    arguments = parser.parse_args()
    main(arguments)
