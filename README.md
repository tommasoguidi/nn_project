# INSTANCE LEVEL RECOGNITION
Obiettivo di questo progetto è analizzare i limiti delle CNN
relativamente al problema del riconoscimento della particolare
istanza di un oggetto. Il dataset scelto è l'Amazon Berkeley
Objects dataset, costituito da inserzioni di prodotti Amazon.
Il dataset è ricco di prodotti molto simili tra loro ma pur
sempre diversi e si presta molto bene al task in questione.

## Data Preprocessing
Il dataset utilizzato è la versione small, contenente immagini
rimpicciolite affinché la dimensione massima non superasse 256
pixel. Il file `images.csv` riporta valori sbagliati delle dimensioni delle
immagini. La correzione è un processo lungo che richiede l'apertura di
tutte le immagini, pertanto è stata fatta una volta per tutte usando
lo script `fix_csv.py` che prende come argomento la `--file-dir`
del file e la `--imgs-dir` contenente le immagini, lo corregge e lo
salva come `fixed_images.csv`.

Di seguito la lista delle operazioni di preprocessing
svolte da `prepare_dataset.py` con i rispettivi parametri che possono
essere fissati da linea di comando:
- scartate le immagini con dimensione massima superiore a `--ratio 
(default = 5)` volte la minima;
- zero-padding delle immagini non quadrate, ottenuto in realtà
aggiungendo pixel bianchi dato che molte delle immagini sono
fotografie da studio su fondo bianco;
- resize a `--size (default = 224)` per ottenere immagini quadrate;
- scartate le inserzioni con meno di `--min-images (default = 5)`
immagini per poter eseguire cross validation;
- selezionate solo le `--n-cat (default = 50)`
categorie merceologiche più rappresentate;
- nella cartella `--dest` vengono salvate le immagini (sotto `images/`)
e il file `subset_images.csv` con le annotazioni relative soltanto alle 
immagini rimaste;
- tramite `--root` è possibile specificare la cartella nella quale sono
stati memorizzati tutti i file così come scaricati dal sito del dataset.

Per quanto riguarda i metadati, questi contengono molti campi, dei
quali ho conservato solo quelli d'interesse per il progetto, ossia
`item_id`, `product_type`, `image_path`, impostando come indice del
dataframe l'`image_id`(che nei metadati originali era conservato
nei campi `main_image_id` e `other_image_id`)