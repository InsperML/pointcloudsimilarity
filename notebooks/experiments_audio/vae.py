import math
import insperdatasets.audio.utils as audio_utils
from insperdatasets.core.datasets import FileLoadingDataset
import insperdatasets.audio.fma as fma
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from torch import nn
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import time

class ConvolutionalVAE(nn.Module):

    def __init__(
        self,
        n_samples: int = 48000,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 256,
        latent_dim: int = 128,
        n_steps: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.inverse_mel = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Determine exact mel spatial dims from a dummy forward pass
        with torch.no_grad():
            mel_dummy = self.mel_transform(torch.zeros(1, 1, n_samples))
            self.mel_h = mel_dummy.shape[-2]  # n_mels
            self.mel_w = mel_dummy.shape[-1]  # n_frames

        # Channel schedule: 1 → C → 2C → ... → C·2^(N-1)
        enc_channels = [1] + [base_channels * (2**i) for i in range(n_steps)]

        # Encoder: N × (Conv2d stride-2 → BN → LeakyReLU)
        enc_layers = []
        for i in range(n_steps):
            enc_layers += [
                nn.Conv2d(enc_channels[i],
                          enc_channels[i + 1],
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(enc_channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Spatial size after encoding: Conv2d(stride=2, pad=1, k=3) maps H → ceil(H/2)
        h_enc, w_enc = self.mel_h, self.mel_w
        for _ in range(n_steps):
            h_enc = math.ceil(h_enc / 2)
            w_enc = math.ceil(w_enc / 2)
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.enc_ch = enc_channels[-1]
        flat_size = self.enc_ch * h_enc * w_enc

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_size)

        # Decoder: N × (ConvTranspose2d stride-2 → BN → LeakyReLU), last step → Sigmoid
        dec_channels = enc_channels[::-1]
        dec_layers = []
        for i in range(n_steps):
            is_last = i == n_steps - 1
            dec_layers.append(
                nn.ConvTranspose2d(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ))
            if is_last:
                dec_layers.append(nn.Sigmoid())
            else:
                dec_layers += [
                    nn.BatchNorm2d(dec_channels[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
        self.decoder_conv = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_samples] → [B, 1, n_mels, n_frames], log1p-scaled"""
        return torch.log1p(self.mel_transform(x))

    def _from_mel_to_audio(self, mel_log: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_mels, n_frames] → [B, n_samples] via Griffin-Lim"""
        mel = torch.expm1(mel_log).clamp(min=0).squeeze(1)
        linear = self.inverse_mel(mel)
        return self.griffin_lim(linear)

    def encode(self, mel: torch.Tensor):
        """[B, 1, n_mels, n_frames] → (mu, logvar) each [B, latent_dim]"""
        h = self.encoder_conv(mel).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] → [B, 1, n_mels, n_frames]"""
        h = self.fc_decode(z).view(-1, self.enc_ch, self.h_enc, self.w_enc)
        out = self.decoder_conv(h)
        # Crop to exact mel size (ConvTranspose2d may overshoot by 1 px)
        return out[:, :, :self.mel_h, :self.mel_w]

    def forward(self, x: torch.Tensor):
        """[B, 1, n_samples] → (recon_mel, mel, mu, logvar)"""
        mel = self._to_mel(x)
        mu, logvar = self.encode(mel)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mel, mu, logvar

    def reconstruct_audio(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_samples] → [B, n_samples] reconstructed via Griffin-Lim"""
        self.eval()
        with torch.no_grad():
            recon_mel, _, _, _ = self.forward(x)
            return self._from_mel_to_audio(recon_mel)


# ------------------------------------------------------------------
# Loss


def vae_loss(recon_mel, mel, mu, logvar, kl_weight: float = 1.0):
    recon = F.mse_loss(recon_mel, mel)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


# ------------------------------------------------------------------
# Training loop


def train_vae(
    model: ConvolutionalVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        k: []
        for k in ("train_loss", "val_loss", "train_recon", "val_recon",
                  "train_kl", "val_kl")
    }

    timestamp = int(time.time())
    run_id = f"vae_epochs{n_epochs}_lr{lr}_kl{kl_weight}_{timestamp}"

    for epoch in tqdm(range(n_epochs),
                      desc="Training VAE",
                      position=0,
                      leave=True):
        model.train()
        t = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for batch in tqdm(train_loader,
                          desc="Train batches",
                          position=1,
                          leave=False):
            audio_data, mask, label = batch
            x = audio_data.to(device)
            optimizer.zero_grad()
            recon_mel, mel, mu, logvar = model(x)
            loss, recon, kl = vae_loss(recon_mel, mel, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            t["loss"] += loss.item()
            t["recon"] += recon.item()
            t["kl"] += kl.item()

        n = len(train_loader)
        history["train_loss"].append(t["loss"] / n)
        history["train_recon"].append(t["recon"] / n)
        history["train_kl"].append(t["kl"] / n)

        model.eval()
        v = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        with torch.no_grad():
            for batch in tqdm(
                    val_loader,
                    desc="Val batches",
                    position=1,
                    leave=False,
            ):
                audio_data, mask, label = batch
                x = audio_data.to(device)
                recon_mel, mel, mu, logvar = model(x)
                loss, recon, kl = vae_loss(recon_mel, mel, mu, logvar,
                                           kl_weight)
                v["loss"] += loss.item()
                v["recon"] += recon.item()
                v["kl"] += kl.item()

        n = len(val_loader)
        history["val_loss"].append(v["loss"] / n)
        history["val_recon"].append(v["recon"] / n)
        history["val_kl"].append(v["kl"] / n)

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"train loss={history['train_loss'][-1]:.4f} "
            f"(recon={history['train_recon'][-1]:.4f} kl={history['train_kl'][-1]:.4f}) | "
            f"val loss={history['val_loss'][-1]:.4f} "
            f"(recon={history['val_recon'][-1]:.4f} kl={history['val_kl'][-1]:.4f})"
        )
        torch.save(model.state_dict(), f'vae_model_weights_{run_id}.pt')

    return history


def main():
    AUDIO_DURATION = 3
    SAMPLE_RATE = 16000
    N_SAMPLES = AUDIO_DURATION * SAMPLE_RATE

    dataset = fma.FMADataset(
        data_dir='/mnt/data3/fma/fma',
        loader_func=partial(
            audio_utils.load_audio,
            crop='random',
            t_len=N_SAMPLES,
        ),
    )

    # Downsample to 1/100 for faster tests
    #dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), 100))
    

    print(f'We have {len(dataset)} tracks in the dataset.')

    # Split dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=40,
        prefetch_factor=2,
        collate_fn=audio_utils.collate_audio,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=40,
        prefetch_factor=2,
        collate_fn=audio_utils.collate_audio,
    )

    model = ConvolutionalVAE(
        n_samples=N_SAMPLES,
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        latent_dim=128,
        n_steps=4,
        base_channels=32,
    ).to('cuda')

    train_vae(
        model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        n_epochs=200,
        lr=1e-4,
        kl_weight=1.0,
        device='cuda',
    )
    
    
    
    # for batch in tqdm(dataloader):
    #     audio_data, mask, label = batch
    #     audio_data = audio_data.to('cuda')
    #     recon_mel, mel, mu, logvar = model(audio_data)
    #print(f"mel: {mel.shape}, recon: {recon_mel.shape}, mu: {mu.shape}")
    #break


if __name__ == "__main__":
    main()
