import torch
from torch import nn 

# Taking input image -> Hidden layer -> Mean and Std dev -> Use reparameterisation trick -> Decoder > Output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_layer=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_size, hidden_layer)
        self.hid_2mu = nn.Linear(hidden_layer, z_dim) # Both layers are separate from eachother and do not feed into eachother
        self.hid_2sigma = nn.Linear(hidden_layer, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, hidden_layer)
        self.hid_2img = nn.Linear(hidden_layer, input_size)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h) # do not use relu here as we want potentially negative values for mu and sigma

        return mu, sigma
    
    def decode(self, z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h)) # values need to be between 0 and 1

    def forward(self,x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterised = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparameterised)
        return x_reconstructed, mu, sigma 

if __name__ == "__main__":
        x = torch.randn(4, 28*28)
        vae = VariationalAutoEncoder(input_size=784)
        x_reconstructed, mu, sigma = vae(x)
        print(x_reconstructed.shape)
        print(mu.shape)
        print(sigma.shape)
        