# intelligenza-computazionale-decadimento


## Ricostruzione del decadimento radioattivo con una rete neurale

### Descrizione del progetto
Questo progetto simula il decadimento radioattivo di una popolazione di atomi e utilizza una rete neurale per stimare la probabilità che un atomo sia ancora integro a un determinato istante.

L'obiettivo è confrontare la curva stimata dalla rete neurale con la curva teorica del decadimento radioattivo, osservando quanto la previsione del modello si avvicini al comportamento previsto dalla legge fisica.

Il progetto è realizzato in Python e utilizza PyTorch per costruire e addestrare la rete neurale.

### Modello fisico
Il decadimento radioattivo viene descritto dalla legge esponenziale:

\[N(t) = N_0 \cdot e^{-\lambda t}\]

dove:
* **N(t)** è il numero medio di atomi superstiti al tempo t;
* **N₀** è il numero iniziale di atomi;
* **t** è il tempo trascorso;
* **λ** è la costante di decadimento.

La costante di decadimento viene calcolata utilizzando il tempo di dimezzamento:

\[\lambda = \frac{\ln(2)}{T_{1/2}}\]

Il tempo di dimezzamento \(T_{1/2}\) rappresenta il tempo necessario affinché il numero medio di atomi radioattivi non ancora decaduti si riduca alla metà.

### Simulazione dei dati
I tempi di decadimento dei singoli atomi vengono generati utilizzando una distribuzione esponenziale.

Per ogni atomo vengono considerati diversi istanti temporali. L'atomo viene classificato come:
* **1**: ancora integro;
* **0**: decaduto.

I dati vengono suddivisi in tre gruppi:
* **Training set (60%)**: utilizzato per addestrare la rete neurale;
* **Validation set (20%)**: utilizzato per monitorare l'errore durante l'addestramento;
* **Test set (20%)**: utilizzato per valutare il modello su dati non impiegati nell'addestramento.

La suddivisione viene effettuata a livello di atomo, in modo che le osservazioni dello stesso atomo non siano distribuite tra gruppi differenti.

### Architettura della rete neurale
La rete neurale è realizzata utilizzando PyTorch e contiene:
* uno strato di ingresso con un valore temporale;
* un primo strato nascosto con 32 neuroni;
* un secondo strato nascosto con 16 neuroni;
* uno strato di uscita con un singolo valore.

Negli strati nascosti viene utilizzata la funzione di attivazione ReLU.

L'uscita della rete è un logit, che viene convertito in una probabilità tramite la funzione sigmoide.

La probabilità stimata rappresenta la probabilità che un atomo sia ancora integro al tempo considerato.

### Addestramento
La funzione di errore utilizzata è `BCEWithLogitsLoss`, adatta a problemi di classificazione binaria.

L'ottimizzatore utilizzato è SGD (Stochastic Gradient Descent), che aggiorna i parametri della rete sulla base dei gradienti dell'errore.

Durante ogni epoca, il modello esegue una previsione, calcola la funzione di errore, determina i gradienti tramite backpropagation e aggiorna i propri parametri.

In questa implementazione, ogni aggiornamento utilizza l'intero training set (Full Batch).

### Metriche di valutazione
Per valutare la classificazione vengono calcolate le seguenti metriche:
* **Accuracy**: proporzione di classificazioni corrette;
* **Precision**: proporzione di veri positivi tra le osservazioni classificate come positive;
* **Recall**: proporzione di veri positivi riconosciuti dal modello;
* **F1-score**: media armonica di Precision e Recall;
* **AUC-ROC**: misura della capacità di distinguere le due classi considerando soglie di classificazione differenti.

L'AUC-ROC viene stimata numericamente utilizzando 101 soglie.

Per confrontare la curva ricostruita con quella teorica vengono inoltre calcolati:
* **RMSE**: radice dell'errore quadratico medio;
* **MAE**: errore assoluto medio.

Queste ultime due metriche misurano la differenza tra i conteggi previsti dalla rete e quelli ottenuti dalla legge teorica.

### Grafico finale
Il programma genera un grafico che mostra:
* i conteggi simulati degli atomi superstiti;
* la curva stimata dalla rete neurale;
* la curva teorica del decadimento radioattivo.

Il grafico viene salvato nel file `risultato_decadimento.png`.

Il confronto permette di valutare visivamente quanto la rete riesca a ricostruire l'andamento esponenziale del fenomeno.

### Requisiti
Per eseguire il progetto sono necessari:
* Python 3
* NumPy
* PyTorch
* Matplotlib

### Installazione
Installa le librerie necessarie dal terminale con il comando:

```bash
python -m pip install numpy torch matplotlib
```

### Esecuzione
> ⚠️ **Nota:** Assicurati di salvare il file di codice con il nome `classificazione_decadimento.py` prima di eseguire i comandi seguenti.

Per avviare il programma con i parametri predefiniti:

```bash
python classificazione_decadimento.py
```

È possibile modificare i parametri direttamente dal terminale:

```bash
python classificazione_decadimento.py --N0 2000 --hl 4 --epoche 500 --lr 0.01
```

I parametri disponibili sono soggetti a regole di validazione interne. Se i valori inseriti non rispettano i vincoli indicati, il programma si interromperà mostrando un errore.

| Parametro | Significato | Valore predefinito | Vincoli di validazione |
| :--- | :--- | :--- | :--- |
| `--N0` | Numero iniziale di atomi | `1500` | Deve essere **almeno 5** |
| `--hl` | Tempo di dimezzamento in secondi | `3.0` | Deve essere **maggiore di 0** |
| `--epoche` | Numero di epoche di addestramento | `300` | Deve essere **almeno 1** |
| `--lr` | Learning rate di SGD | `0.01` | Deve essere **maggiore di 0** |

Al termine dell'esecuzione vengono stampate le metriche di classificazione e gli errori della curva. Viene inoltre salvato il grafico `risultato_decadimento.png`.

### Obiettivo
Il progetto permette di approfondire la simulazione numerica di un fenomeno fisico, la classificazione binaria tramite una rete neurale, il processo di addestramento con SGD e il confronto tra una previsione numerica e un modello teorico.

Il lavoro mostra come una rete neurale possa essere utilizzata per stimare il comportamento medio di un sistema fisico a partire da dati simulati.
