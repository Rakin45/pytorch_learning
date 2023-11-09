import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-4 

# Dataset loading 
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.BCELoss(reduction="sum")

# Training
train_losses = []
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (x, _) in loop:
        x = x.view(x.size(0), -1).to(DEVICE)
        x_reconstructed, mu, sigma = model(x)
        # Reconstruction loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        # KL divergence loss
        kl_divergence_loss = 0.5 * torch.sum(torch.exp(sigma) + mu**2 - 1.0 - sigma)
        # Total loss
        total_loss = reconstruction_loss + kl_divergence_loss
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Logging
        train_losses.append(total_loss.item())
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=total_loss.item())
    
import os

model = model.to("cpu")

def inference(digit, num_samples=5):
    images = []
    idx = 0
    for x, y in dataset:
        if y == digit:
            images.append(x)
            idx += 1
        if idx == num_samples:
            break
    encoding_digits = []
    with torch.no_grad():
        mu, sigma = model.encode(torch.stack(images).view(-1, 784))
        encoding_digits.append((mu, sigma))
    return encoding_digits

# Create the results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

num_samples = 5
for digit in range(10):
    encoding_digits = inference(digit, num_samples)
    mu, sigma = encoding_digits[0]
    for example in range(num_samples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"results/{digit}_{example}.png")
