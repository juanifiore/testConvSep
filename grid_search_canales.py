import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import time

sr = 44100
n_fft = 1024
hop = 256
frame_len = 256  # número de frames

mix_path = "../hendrix/Music_Delta_-_Hendrix_recortado.stem_mix.wav"
stem_path = "../hendrix/Music_Delta_-_Hendrix_recortado.stem_vocals.wav" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def match_rms(y_pred, y_target):
    rms_pred = np.sqrt(np.mean(y_pred**2))
    rms_target = np.sqrt(np.mean(y_target**2))
    return y_pred * (rms_target / (rms_pred + 1e-8))



def load_and_chunks(path_mix, path_target):
    mix, _ = librosa.load(path_mix, sr=sr, mono=True)
    tgt, _ = librosa.load(path_target, sr=sr, mono=True)
    S_mix = np.abs(librosa.stft(mix, n_fft=n_fft, hop_length=hop))
    S_tgt = np.abs(librosa.stft(tgt, n_fft=n_fft, hop_length=hop))
    S_mix = np.log1p(S_mix)
    S_tgt = np.log1p(S_tgt)
    #S_mix = 10 * np.log10(S_mix + 1e-10)
    #S_tgt = 10 * np.log10(S_tgt + 1e-10)

    chunks, masks = [], []
    for i in range(0, S_mix.shape[1]-frame_len, frame_len):
        x = S_mix[:, i:i+frame_len]
        y = S_tgt[:, i:i+frame_len]
        m = np.clip(y / (x+1e-6), 0., 1.)
        chunks.append(x)
        masks.append(m)
    #X = np.stack(chunks)[..., None]
    #M = np.stack(masks)[..., None]
    X = np.stack(chunks)[:, None, :, :]  # (N, 1, 513, 256)
    M = np.stack(masks)[:, None, :, :]   # (N, 1, 513, 256)

    return torch.from_numpy(X).float(), torch.from_numpy(M).float()

X, M = load_and_chunks(mix_path, stem_path)
dataset = DataLoader(torch.utils.data.TensorDataset(X, M), batch_size=8, shuffle=True)


class FlexibleCNN(nn.Module):
    def __init__(self, num_layers=3, kernel_sizes=None, strides=None, input_channels=1, output_channels=16):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers
        if strides is None:
            strides = [1] * num_layers

        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers

        layers = []
        in_channels = input_channels
        for i in range(num_layers):
            #out_channels = 16 * (2 ** i)
            out_channels = output_channels
            k = kernel_sizes[i]
            s = strides[i]
            p = k // 2  # padding para mantener tamaño si stride=1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.cnn(x)
        mask = torch.sigmoid(self.final_conv(x))
        return mask



#class SimpleCNN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
#        self.bn1 = nn.BatchNorm2d(16)
#        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
#        self.bn2 = nn.BatchNorm2d(32)
#        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#        self.bn3 = nn.BatchNorm2d(64)
#        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)
#
#    def forward(self, x):
#        #x0 = x
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
#        mask = torch.sigmoid(self.conv4(x))
#        return mask 


configs = [
    {'num_layers': 4, 'kernel_sizes': [5, 5, 5, 5], 'strides': [1, 1, 1, 1], 'canales': 16},
    {'num_layers': 4, 'kernel_sizes': [5, 5, 5, 5], 'strides': [1, 1, 1, 1], 'canales': 32},
    {'num_layers': 4, 'kernel_sizes': [5, 5, 5, 5], 'strides': [1, 1, 1, 1], 'canales': 64},
    {'num_layers': 4, 'kernel_sizes': [5, 5, 5, 5], 'strides': [1, 1, 1, 1], 'canales': 128},
    {'num_layers': 4, 'kernel_sizes': [5, 5, 5, 5], 'strides': [1, 1, 1, 1], 'canales': 256},
]


loses_todas = []
criterion = nn.MSELoss()
for i, config in enumerate(configs):
    print(f"Config {i+1}: kernels={config['kernel_sizes']}, strides={config['strides']}")
    model = FlexibleCNN(
        num_layers=config['num_layers'],
        kernel_sizes=config['kernel_sizes'],
        strides=config['strides'],
        output_channels=config['canales']
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loses = []
    start_time = time.time()
    for epoch in range(150):
        total_loss = 0.0
        for xb, mb in dataset:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            loses.append(loss.item())
        print(f"Epoch {epoch+1}, loss: {total_loss/len(dataset):.4f}")

    # predecir y reconstruir sobre los mismos chunks
    with torch.no_grad():
        model.eval()
        pred = model(X.to(DEVICE)).cpu().detach().numpy()[:, 0, :, :]

    mask = pred
    S_mix_all = np.log1p(np.abs(librosa.stft(librosa.load(mix_path, sr=sr, mono=True)[0],
                                              n_fft=n_fft, hop_length=hop)))
    # reconstrucción por bloques
    reconstructed = []
    for j in range(mask.shape[0]):
        mm = mask[j]
        sm = np.expm1(S_mix_all[:, j*frame_len:(j*frame_len+frame_len)])
        recon = mm * sm
        #recon = mm 
        reconstructed.append(recon)
    S_rec = np.concatenate(reconstructed, axis=1)
    # ISTFT
    mix, _ = librosa.load(mix_path, sr=sr, mono=True)

    y = librosa.istft(S_rec, hop_length=hop, length=mix.shape[0])

    end_time = time.time()
    duration = end_time - start_time
    np.save(f'../hendrix/grid/time_config{i+1}_chanels{config["canales"]}.npy', np.array([duration]))

    sf.write(f'../hendrix/grid/vocals_config{i+1}_chanels{config["canales"]}.wav', y, sr)
    #loses_todas.append(loses)
    np.save(f'../hendrix/grid/loss_config{i+1}_chanels{config["canales"]}.npy', loses)
    np.save(f'../hendrix/grid/mask_pred_config{i+1}_chanels{config["canales"]}.npy', pred)

    del model
    torch.cuda.empty_cache()


np.save(f'../hendrix/grid/mask_true.npy', M.numpy())  # solo una vez
#np.save('loses_grid.npy', loses_todas)


