# INSTANCE LEVEL RECOGNITION
Obiettivo di questo progetto è analizzare i limiti delle CNN
relativamente al problema del riconoscimento della particolare
istanza di un oggetto. Il dataset scelto è l'Amazon Berkeley
Objects dataset, costituito da inserzioni di prodotti Amazon.
Il dataset è ricco di prodotti molto simili tra loro ma pur
sempre diversi e si presta molto bene al task in questione.

## Dependencies
Il progetto utilizza solo librerie molto comuni, come pytorch, numpy e
pandas. L'elaborazione delle immagini è stata eseguita usando opencv. In
ogni caso i file `requirements.txt` e `environment.yaml` permettono di
installare tutto quello che serve per far girare il codice, che usa la
versione 3.10 di Python.

## Data Preprocessing
Il dataset utilizzato è la versione small, contenente immagini
rimpicciolite affinché la dimensione massima non superasse 256
pixel. Il file `images.csv` riporta valori sbagliati delle dimensioni delle
immagini. La correzione è un processo lungo che richiede l'apertura di
tutte le immagini, pertanto è stata fatta una volta per tutte usando
lo script `fix_csv.py`.

### `fix_csv.py`
Lo script prende come argomento la `--file-dir`
del file e la `--imgs-dir` contenente le immagini, lo corregge e lo
salva come `fixed_images.csv`.

### `prepare_dataset.py`
Questo script svolge le operazioni di preprocessing del dataset per estrarne
un subset. Inizialmente volevo tenere le 50 classi più rappresentate, poi
per motivi di tempo ho deciso di limitarlo a 10. Eseguendo lo script si 
ottiene il file `subset_images.csv` con le sole annotazioni utili al
progetto, relative alle immagini selezionate nel subset. Al tempo stesso
le immagini vengono rese quadrate mediante padding con pixel bianchi, poi
ridimensionate alla dimensione desiderata e infine salvate nella directory
indicata.

Di seguito nel dettaglio la lista delle operazioni di preprocessing
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
- selezionate solo le `--n-cat (default = 10)`
categorie merceologiche più rappresentate;
- per bilanciare un po' il dataset così ottenuto, si fa un subsampling
sui prodotti (così da non eliminare eventualmente immagini, il che avrebbe
potuto portare a inserzioni con meno di `--min-images` immagini). Per farlo,
tutte le classi con un numero di inserzioni superiore alla media `avg`, sono
state campionate con probabilità inversamente proporzionale ad esso;
- nella cartella `--dest` vengono salvate le immagini (sotto `images/`)
e il file `subset_images.csv` con le annotazioni relative soltanto alle 
immagini rimaste;
- tramite `--root` è possibile specificare la cartella nella quale sono
stati memorizzati tutti i file così come scaricati dal sito del dataset.

Per quanto riguarda i metadati, questi contengono molti campi, dei
quali ho conservato solo quelli d'interesse per il progetto, ossia
`item_id`, `product_type`, `image_path`, impostando come indice del
dataframe l'`image_id`(che nei metadati originali era conservato
nei campi `main_image_id` e `other_image_id`).

## Classificazione

La prima fase del progetto si incetra sull'allenamento di un classificatore
in grado di riconoscere il `product_type` tra i 10 presenti nel subset. Per
farlo ho implementato una CNN basilare e l'ho confrontata con una ResNet
preallenata su ImageNet, eseguendo il fine-tuning dei parametri
appartenenti all'ultimo bottleneck.

### `classifier.py`

Di seguito gli argomenti che possono essere passati allo script da linea di
comando:
- `--root` è la cartella che contiene il `.csv` delle annotazioni e
le immagini nelle sottocartelle originarie, sotto la cartella `images/`;
- `--n-folds` è il numero di fold per la k-fold validation;
- `--mode` per decidere se si vuole eseguire il 'train' o il 'test' della 
rete;
- `--device` se usare la cpu o la gpu;
- `--backbone` se usare la CNN o la ResNet;
- `--epochs` il numero di epoche per cui eseguire l'allenamento;
- `--batch-size` la dimensione del batch in fase di training;
- `--num-workers` per la parallelizzazione;
- `--lr` il learning rate iniziale;
- `--checkpoint_dir` la cartella in cui salvare il summary dell'allenamento
e i pesi del modello selezionato in base all'acuratezza sul validation;
- `--weights` il file dei pesi da caricare qualora si volesse valutare il
modello sul dataset di test.

## Instance Recognition

La seconda fase del progetto si incentra sul tentativo di sviluppare un
modello in grado non più di classificare gli oggetti secondo il loro
`product_type`, ma di riconoscere i singoli prodotti all'interno di
ciascuna categoria.
