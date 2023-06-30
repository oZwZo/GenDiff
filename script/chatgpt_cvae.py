import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Load single-cell expression data and metadata into a scanpy AnnData object
adata = sc.read_h5ad("path/to/data.h5ad")

# Preprocess the data
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat")
adata.raw = adata
adata = adata[:, adata.var["highly_variable"]]

# Split the data into training and testing sets
adata_train, adata_test = sc.model.zoo.split_setup(adata)

class ConditionedDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata

    def __getitem__(self, index):
        x = self.adata[index].X.toarray().astype(np.float32)
        c = self.adata[index].obs[condition_col].values.astype(np.float32)
        return x, c

    def __len__(self):
        return len(self.adata)

# Define the VAE model architecture
class VAE(nn.Module):
    def __init__(self, n_input, n_condition, n_hidden=128, n_latent=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input + n_condition, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latent + n_condition, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_input)
        )

    def encode(self, x, c):
        input_tensor = torch.cat([x, c], dim=1)
        mu_logvar = self.encoder(input_tensor)
        mu = mu_logvar[:, :n_latent]
        logvar = mu_logvar[:, n_latent:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, c):
        input_tensor = torch.cat([z, c], dim=1)
        output = self.decoder(input_tensor)
        return output

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, c)
        return output, mu, logvar

# Train the VAE model
batch_size = 256
lr = 1e-3
n_epochs = 100

condition_col = "Condition"
n_input = adata_train.n_vars
n_condition = adata_train.obs[condition_col].nunique()
n_latent = 10

train_dataset = ConditionedDataset(adata_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

vae = VAE(n_input=n_input, n_condition=n_condition, n_latent=n_latent)
optimizer = optim.Adam(vae.parameters(), lr=lr)

for epoch in range(n_epochs):
    vae.train()
    train_loss = 0.0
    for x, c in train_loader:
        optimizer.zero_grad()

        x_hat, mu, logvar = vae(x, c)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kl_divergence) / x.shape[0]

        loss.backward()
        train_loss += loss.item() * x.shape[0]
        optimizer.step()

    train_loss /= len(train_dataset)

    print("Epoch: {}, Train Loss: {}".format(epoch+1, train_loss))

# Generate new samples from specific conditions
def generate_samples(vae, n_samples, condition):
    vae.eval()

    c = np.zeros((n_samples, n_condition))
    c[:, condition] = 1

    c_tensor = torch.from_numpy(c).float().cuda()

    z = torch.randn((n_samples, n_latent))
    z_tensor = z.float().cuda()

    with torch.no_grad():
        x_tensor = vae.decode(z_tensor, c_tensor)

    return x_tensor.cpu().numpy()


# Generate 10 samples from condition 0
samples = generate_samples(vae, n_samples=10, condition=0)

# Evaluate the performance of the model on the test set
test_dataset = ConditionedDataset(adata_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

vae.eval()
test_loss = 0.0
for x, c in test_loader:
    x_hat, mu, logvar = vae(x, c)
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (recon_loss + kl_divergence) / x.shape[0]
    test_loss += loss.item() * x.shape[0]

test_loss /= len(test_dataset)

print("Test Loss: {}".format(test_loss))



## In this script, the `ConditionedDataset` class defines a PyTorch dataset that can be used for training and testing the VAE model. The `VAE` class defines the architecture of the model, including the encoder and decoder networks.
## The `generate_samples` function takes in a trained VAE model, the number of samples to generate, and a specific condition for which to generate the samples. The function returns a numpy array containing the generated samples.
## Finally, the script evaluates the performance of the model on the test set by computing the reconstruction loss and KL divergence between the reconstructed data and the original data.
## Note that the script assumes that the input data is stored in an AnnData object with metadata containing a column for experimental conditions. The actual implementation may need to be modified based on the specific format of your data.