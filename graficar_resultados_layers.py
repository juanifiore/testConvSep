import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Ruta a la carpeta donde están los archivos
base_path = "hendrix/grid"

# Configuraciones que usaste
configs = [
    {'config_id': 1, 'num_layers': 2},
    {'config_id': 2, 'num_layers': 3},
    {'config_id': 3, 'num_layers': 4},
    {'config_id': 4, 'num_layers': 5},
    {'config_id': 5, 'num_layers': 6},
]

# --- 1. Evolución de la loss ---
plt.figure(figsize=(10, 6))

for cfg in configs:
    config_id = cfg['config_id']
    num_layers = cfg['num_layers']
    loss_path = os.path.join(base_path, f"loss_config{config_id}_layers{num_layers}.npy")
    if os.path.exists(loss_path):
        loss = np.load(loss_path)
        plt.plot(loss, label=f"{num_layers} capas")
    else:
        print(f"[WARN] No se encontró: {loss_path}")

plt.title("Evolución de la pérdida por época")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.tight_layout()
plt.savefig("loss_por_epoch.png")
# plt.show()  ← esto lo eliminás si no querés mostrarlo
plt.close()


# --- 2. Curva con tiempos de ejecución ---
tiempos = []
for cfg in configs:
    config_id = cfg['config_id']
    num_layers = cfg['num_layers']
    time_path = os.path.join(base_path, f"time_config{config_id}_layers{num_layers}.npy")
    if os.path.exists(time_path):
        tiempo = np.load(time_path)[0]
        tiempos.append({'Capas': num_layers, 'Tiempo (segundos)': tiempo})
    else:
        print(f"[WARN] No se encontró: {time_path}")

# Si hay datos, graficar
if tiempos:
    df_tiempos = pd.DataFrame(tiempos).sort_values(by="Capas")
    
    # Mostrar como tabla en consola
    print("\n📊 Tiempos de ejecución:")
    print(df_tiempos.round(2).to_string(index=False))
    
    # Graficar curva
    plt.figure(figsize=(8, 5))
    plt.plot(df_tiempos["Capas"], df_tiempos["Tiempo (segundos)"], marker='o', linestyle='-')
    plt.title("Tiempo de ejecución vs cantidad de capas")
    plt.xlabel("Cantidad de capas")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tiempos_vs_capas.png", dpi=150)
    plt.close()
    print("✅ Gráfico guardado como 'tiempos_vs_capas.png'")
else:
    print("⚠️ No se encontraron datos de tiempo para graficar.")


# --- 3. Gráfico de máscaras predichas vs verdadera ---
# Cargar máscara verdadera
mask_true_path = os.path.join(base_path, "mask_true.npy")
if not os.path.exists(mask_true_path):
    print("\n[ERROR] No se encontró 'mask_true.npy'")
else:
    mask_true = np.load(mask_true_path)
    n = 0  # índice del ejemplo a mostrar

    # Preparar figura
    fig, axes = plt.subplots(1, len(configs) + 1, figsize=(18, 4))

    # Máscaras predichas
    for idx, cfg in enumerate(configs):
        config_id = cfg['config_id']
        num_layers = cfg['num_layers']
        pred_path = os.path.join(base_path, f"mask_pred_config{config_id}.npy")
        if os.path.exists(pred_path):
            mask_pred = np.load(pred_path)
            axes[idx].imshow(mask_pred[n], aspect='auto', origin='lower')
            axes[idx].set_title(f"Predicción\n{num_layers} capas")
            axes[idx].axis('off')
        else:
            axes[idx].set_title(f"No pred.\n{num_layers} capas")
            axes[idx].axis('off')

    # Máscara verdadera
#    axes[-1].imshow(mask_true[n], aspect='auto', origin='lower')
    axes[-1].imshow(mask_true[n].squeeze(), aspect='auto', origin='lower')

    axes[-1].set_title("Máscara verdadera")
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.savefig("mascaras_predichas_vs_true.png")
    # plt.show()
    plt.close()


