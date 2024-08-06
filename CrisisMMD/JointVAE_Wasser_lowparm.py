import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, input_size):
        super(FusionLayer, self).__init__()
        self.wc = nn.Linear(in_features=input_size, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x1 = x1.unsqueeze(dim=1)
        x2 = x2.unsqueeze(dim=1)
        x = torch.cat((x1, x2), dim=1)
        xc = self.softmax(self.wc(x)).permute(0, 2, 1)
        out = torch.matmul(xc, x).squeeze()
        output = self.tanh(out)
        return output


class JointVAE(nn.Module):
    def __init__(self, input_size_A, input_size_B, input_size, seq_len):
        super(JointVAE, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len

        self.input_size_A = input_size_A
        self.input_size_B = input_size_B

        self.en_fc_B = nn.Linear(in_features=self.input_size_B // 4, out_features=(input_size * seq_len) // 4)

        self.encoder_A = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),

            nn.Conv1d(in_channels=input_size * 2, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.encoder_linear_A = nn.Sequential(
            nn.Linear(in_features=input_size * 2, out_features=input_size)
        )

        self.encoder_B = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),

            nn.Conv1d(in_channels=input_size * 2, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.encoder_linear_B = nn.Sequential(
            nn.Linear(in_features=input_size * 2, out_features=input_size)
        )

        self.fc_mu = FusionLayer(input_size=input_size * seq_len)
        self.fc_logvar = FusionLayer(input_size=input_size * seq_len)

        self.decoder_A = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_size, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose1d(in_channels=input_size * 2, out_channels=input_size * 2, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.decoder_linear_A = nn.Sequential(
            nn.Linear(in_features=(input_size * seq_len) // 2, out_features=(input_size * seq_len) // 4)
        )

        self.decoder_B = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_size, out_channels=input_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),

            nn.ConvTranspose1d(in_channels=input_size * 2, out_channels=input_size * 2, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm1d(num_features=input_size * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.decoder_linear_B = nn.Sequential(
            nn.Linear(in_features=(input_size * seq_len) // 2, out_features=self.input_size_B // 4)
        )

    def encode(self, x_A, x_B):
        x_B = x_B.view(x_B.shape[0], 4, self.input_size_B // 4)
        x_B = self.en_fc_B(x_B)
        x_A = x_A.view(x_A.shape[0], self.input_size, self.seq_len)
        x_B = x_B.view(x_B.shape[0], self.input_size, self.seq_len)
        h_A = self.encoder_A(x_A)
        h_A = h_A.permute(0, 2, 1)
        h_A = self.encoder_linear_A(h_A)
        h_B = self.encoder_B(x_B)
        h_B = h_B.permute(0, 2, 1)
        h_B = self.encoder_linear_B(h_B)
        mu = self.fc_mu(h_A, h_B)
        logvar = self.fc_logvar(h_A, h_B)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_l):
        x_A_reconstructed = self.decoder_A(z_l)
        x_A_reconstructed = x_A_reconstructed.view(x_A_reconstructed.shape[0], 4, (self.input_size * self.seq_len) // 2)
        x_A_reconstructed = self.decoder_linear_A(x_A_reconstructed)
        x_B_reconstructed = self.decoder_B(z_l)
        x_B_reconstructed = x_B_reconstructed.view(x_B_reconstructed.shape[0], 4, (self.input_size * self.seq_len) // 2)
        x_B_reconstructed = self.decoder_linear_B(x_B_reconstructed)
        x_A_reconstructed = x_A_reconstructed.view(x_A_reconstructed.shape[0], -1)
        x_B_reconstructed = x_B_reconstructed.view(x_B_reconstructed.shape[0], -1)
        return x_A_reconstructed, x_B_reconstructed

    def forward(self, x_A, x_B):
        mu, logvar = self.encode(x_A, x_B)
        z = self.reparameterize(mu, logvar)
        z_l = z.view(z.shape[0], self.input_size, self.seq_len)
        x_A_reconstructed, x_B_reconstructed = self.decode(z_l)
        return z, x_A_reconstructed, x_B_reconstructed, mu, logvar
