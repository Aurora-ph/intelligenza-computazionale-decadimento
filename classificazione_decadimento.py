import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Simulazione del decadimento radioattivo"
)

parser.add_argument(
    "--N0", type=int, default=1500,
    help="Numero iniziale di atomi"
)

parser.add_argument(
    "--hl", type=float, default=3.0,
    help="Tempo di dimezzamento in secondi"
)

parser.add_argument(
    "--epoche", type=int, default=300,
    help="Numero di epoche"
)

parser.add_argument(
    "--lr", type=float, default=0.01,
    help="Learning rate dell'ottimizzatore SGD"
)

args = parser.parse_args()

N0 = args.N0
tempo_dimezzamento = args.hl
epoche = args.epoche
learning_rate = args.lr

if N0 < 5 or tempo_dimezzamento <= 0 or epoche < 1 or learning_rate <= 0:
    parser.error(
        "N0 deve essere almeno 5; tempo di dimezzamento, "
        "epoche e learning rate devono essere positivi."
    )

np.random.seed(42)
torch.manual_seed(42)

lam = np.log(2) / tempo_dimezzamento

print("=== PARAMETRI DEL PROGETTO ===")
print(f"Numero iniziale di atomi: {N0}")
print(f"Tempo di dimezzamento: {tempo_dimezzamento} secondi")
print(f"Numero di epoche: {epoche}")
print(f"Learning rate SGD: {learning_rate}")


tempi = np.linspace(0, 15, 100)

tempi_decadimento = np.random.exponential(
    scale=1 / lam,
    size=N0
)

indici = np.random.permutation(N0)

n_train = int(0.6 * N0)
n_val = int(0.2 * N0)

atomi_train = indici[:n_train]
atomi_val = indici[n_train:n_train + n_val]
atomi_test = indici[n_train + n_val:]


def crea_dataset(indici_atomi):
    tempi_norm = tempi / 15.0

    X = np.tile(
        tempi_norm,
        len(indici_atomi)
    ).reshape(-1, 1)

    Y = (
        tempi_decadimento[indici_atomi, None]
        > tempi[None, :]
    ).astype(np.float32).reshape(-1, 1)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y


X_train, Y_train = crea_dataset(atomi_train)
X_val, Y_val = crea_dataset(atomi_val)
X_test, Y_test = crea_dataset(atomi_test)

print("\n=== DIMENSIONI DEI DATASET ===")
print(f"Training: {len(X_train)} osservazioni")
print(f"Validation: {len(X_val)} osservazioni")
print(f"Test: {len(X_test)} osservazioni")


class ReteDecadimento(nn.Module):

    def __init__(self):
        super().__init__()

        self.strato1 = nn.Linear(1, 32)
        self.strato2 = nn.Linear(32, 16)
        self.uscita = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.strato1(x))
        x = self.relu(self.strato2(x))
        return self.uscita(x)


model = ReteDecadimento()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)


print("\n=== INIZIO ADDESTRAMENTO ===")

for epoca in range(epoche):

    model.train()

    optimizer.zero_grad()

    logit_train = model(X_train)

    loss_train = criterion(logit_train, Y_train)

    loss_train.backward()

    optimizer.step()

    model.eval()

    with torch.no_grad():
        logit_val = model(X_val)
        loss_val = criterion(logit_val, Y_val)

    if (epoca + 1) % 50 == 0:
        print(
            f"Epoca {epoca + 1}/{epoche} | "
            f"Train loss: {loss_train.item():.4f} | "
            f"Validation loss: {loss_val.item():.4f}"
        )


model.eval()

with torch.no_grad():
    logit_test = model(X_test)
    prob_test = torch.sigmoid(logit_test).numpy().ravel()


y_true = Y_test.numpy().ravel().astype(int)
y_pred = (prob_test >= 0.5).astype(int)

TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

accuracy = (TP + TN) / len(y_true)

precision = TP / (TP + FP) if TP + FP > 0 else 0.0

recall = TP / (TP + FN) if TP + FN > 0 else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0
)

soglie = np.linspace(0, 1, 101)

tpr = []
fpr = []

for soglia in soglie:

    pred = (prob_test >= soglia).astype(int)

    tp = np.sum((y_true == 1) & (pred == 1))
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    tn = np.sum((y_true == 0) & (pred == 0))

    tpr.append(
        tp / (tp + fn) if tp + fn > 0 else 0.0
    )

    fpr.append(
        fp / (fp + tn) if fp + tn > 0 else 0.0
    )


ordine = np.argsort(fpr)
fpr = np.array(fpr)[ordine]
tpr = np.array(tpr)[ordine]

auc_roc = np.sum(
    (fpr[1:] - fpr[:-1]) *
    (tpr[1:] + tpr[:-1]) / 2
)

print("\n=== METRICHE SUL TEST SET ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")


X_grafico = torch.tensor(
    (tempi / 15.0).reshape(-1, 1),
    dtype=torch.float32
)

with torch.no_grad():
    prob_nn = torch.sigmoid(model(X_grafico)).numpy().ravel()

curva_nn = N0 * prob_nn

curva_teorica = N0 * np.exp(-lam * tempi)

conteggio_reale = np.sum(
    tempi_decadimento[:, None] > tempi[None, :],
    axis=0
)

rmse = np.sqrt(
    np.mean((curva_nn - curva_teorica) ** 2)
)

mae = np.mean(
    np.abs(curva_nn - curva_teorica)
)

print("\n=== ERRORI DELLA CURVA ===")
print(f"RMSE: {rmse:.2f} atomi")
print(f"MAE:  {mae:.2f} atomi")


plt.figure(figsize=(10, 6))

plt.scatter(
    tempi,
    conteggio_reale,
    color="crimson",
    alpha=0.5,
    label="Dati simulati"
)

plt.plot(
    tempi,
    curva_nn,
    color="dodgerblue",
    linewidth=2.5,
    label="Previsione della rete"
)

plt.plot(
    tempi,
    curva_teorica,
    color="black",
    linestyle="--",
    label="Legge teorica"
)

plt.xlabel("Tempo (secondi)")
plt.ylabel("Atomi superstiti")
plt.title("Ricostruzione del decadimento radioattivo")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig("risultato_decadimento.png", dpi=150)

plt.show()
