import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Ruta a la carpeta donde están los archivos
base_path = "hendrix/grid"

# Configuraciones por cantidad de canales
configs = [
    {'config_id': 1, 'channels': 16},
    {'config_id': 2, 'channels': 32},
    {'config_id': 3, 'channels': 64},
    {'config_id': 4, 'channels': 128},
    {'config_id': 5, 'channels': 256},
]

# --- 1. Evolución de la loss ---
plt.figure(figsize=(10, 6))

for cfg in configs:
    config_id = cfg['config_id']
    channels = cfg['channels']
    loss_path = os.path.join(base_path, f"loss_config{config_id}_chanels{channels}.npy")
    if os.path.exists(loss_path):
        loss = np.load(loss_path)
        plt.plot(loss, label=f"{channels} canales")
    else:
        print(f"[WARN] No se encontró: {loss_path}")

plt.title("Evolución de la pérdida por época")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_por_epoch_channels.png", dpi=150)
plt.close()


# --- 2. Curva con tiempos de ejecución ---
tiempos = []
for cfg in configs:
    config_id = cfg['config_id']
    channels = cfg['channels']
    time_path = os.path.join(base_path, f"time_config{config_id}_chanels{channels}.npy")
    if os.path.exists(time_path):
        tiempo = np.load(time_path)[0]
        tiempos.append({'Canales': channels, 'Tiempo (segundos)': tiempo})
    else:
        print(f"[WARN] No se encontró: {time_path}")

# Si hay datos, graficar
if tiempos:
    df_tiempos = pd.DataFrame(tiempos).sort_values(by="Canales")
    plt.figure(figsize=(8, 5))
    plt.plot(df_tiempos["Canales"], df_tiempos["Tiempo (segundos)"], marker='o')
    plt.title("Tiempo de ejecución vs cantidad de canales")
    plt.xlabel("Cantidad de canales")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tiempos_vs_channels.png", dpi=150)
    plt.close()
    print("✅ Gráfico guardado como 'tiempos_vs_channels.png'")
else:
    print("⚠️ No se encontraron datos de tiempo para graficar.")


# --- 3. Gráfico de máscaras predichas vs verdadera ---
mask_true_path = os.path.join(base_path, "mask_true.npy")
if not os.path.exists(mask_true_path):
    print("\n[ERROR] No se encontró 'mask_true.npy'")
else:
    mask_true = np.load(mask_true_path)
    n = 0  # índice del ejemplo a mostrar

    fig, axes = plt.subplots(1, len(configs) + 1, figsize=(20, 4))

    for idx, cfg in enumerate(configs):
        config_id = cfg['config_id']
        channels = cfg['channels']
        pred_path = os.path.join(base_path, f"mask_pred_config{config_id}_chanels{channels}.npy")
        if os.path.exists(pred_path):
            mask_pred = np.load(pred_path)
            axes[idx].imshow(mask_pred[n].squeeze(), aspect='auto', origin='lower')
            axes[idx].set_title(f"Predicción\n{channels} canales")
            axes[idx].axis('off')
        else:
            axes[idx].set_title(f"No pred.\n{channels} canales")
            axes[idx].axis('off')

    axes[-1].imshow(mask_true[n].squeeze(), aspect='auto', origin='lower')
    axes[-1].set_title("Máscara verdadera")
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig("mascaras_predichas_vs_true_channels.png", dpi=150)
    plt.close()

