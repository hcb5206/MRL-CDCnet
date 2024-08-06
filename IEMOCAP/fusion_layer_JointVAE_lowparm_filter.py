import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans
from complementarity_learning import ComplementarityAttention
from JointVAE_Wasser_lowparm import JointVAE


def onehotencode(labels, output_size):
    encoded_labels = np.zeros((len(labels), output_size))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels


class Simi_Diff_Fusion(nn.Module):
    def __init__(self, input_size):
        super(Simi_Diff_Fusion, self).__init__()
        self.wc = nn.Linear(in_features=input_size, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x1 = x1.unsqueeze(dim=1)
        x2 = x2.unsqueeze(dim=1)
        x3 = x3.unsqueeze(dim=1)
        x = torch.cat((x1, x2, x3), dim=1)
        xc = self.softmax(self.wc(x)).permute(0, 2, 1)
        out = torch.matmul(xc, x).squeeze()
        output = self.tanh(out)
        return output


class FusionLayerAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super(FusionLayerAttention, self).__init__()
        self.wc = nn.Linear(in_features=input_size, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(num_features=input_size)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, simi, diff, comple_out, speaker):
        simi = simi.unsqueeze(dim=1)
        diff = diff.unsqueeze(dim=1)
        comple_out = comple_out.unsqueeze(dim=1)
        speaker = speaker.unsqueeze(dim=1)
        x = torch.cat((simi, diff, comple_out, speaker), dim=1)
        xc = self.softmax(self.wc(x)).permute(0, 2, 1)
        out = torch.matmul(xc, x).squeeze()
        out = self.bn(out)
        out = self.tanh(out)
        output = self.dropout(out)
        return output


class DiffFilter(nn.Module):
    def __init__(self, seq_len, input_size, p):
        super(DiffFilter, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.p = p
        self.w1 = nn.Linear(in_features=seq_len, out_features=input_size)
        self.w2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, diff):
        diff = diff.view(diff.shape[0], self.seq_len, self.input_size)
        diff_t = diff.transpose(1, 2)
        d_1 = self.w1(diff_t).transpose(1, 2)
        d_2 = self.w2(d_1)
        eps = torch.rand_like(d_2)
        df = self.sigmoid(torch.log(eps / (1 - eps)) + d_2)
        df_z = (df >= self.p).float()
        df_z = df + (df_z - df).detach()
        output = torch.matmul(diff, df_z)
        output = output.view(output.shape[0], -1)
        return output


class OutputLayer(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_heads, dropout, k, label_size, classer,
                 input_size_img, input_size_word, input_size_audio, num_speaker, p):
        super(OutputLayer, self).__init__()
        self.classer = classer
        self.input_size = input_size
        self.seq_len = seq_len
        self.input_size_img = input_size_img
        self.input_size_word = input_size_word
        self.input_size_audio = input_size_audio

        self.jointVAE = JointVAE(input_size_A=input_size_img, input_size_B=input_size_word,
                                 input_size_C=input_size_audio, input_size=input_size, seq_len=seq_len)

        self.normBNa = nn.BatchNorm1d(input_size_word, affine=True)
        self.normBNb = nn.BatchNorm1d(input_size_word, affine=True)
        self.normBNc = nn.BatchNorm1d(input_size_word, affine=True)
        self.normBNd = nn.BatchNorm1d(input_size_word, affine=True)

        self.normBN_simi = nn.BatchNorm1d(input_size * seq_len, affine=True)

        self.fc_img = nn.Linear(in_features=input_size_img // 2, out_features=(input_size * seq_len) // 2)
        self.normBN_img = nn.BatchNorm1d(input_size * seq_len, affine=True)

        self.diff_fc_img = nn.Linear(in_features=input_size_img // 2, out_features=(input_size * seq_len) // 2)
        self.normBN_diff_img = nn.BatchNorm1d(input_size * seq_len, affine=True)

        self.fc_audio = nn.Linear(in_features=input_size_audio // 2, out_features=(input_size * seq_len) // 2)
        self.normBN_audio = nn.BatchNorm1d(input_size * seq_len, affine=True)

        self.diff_fc_audio = nn.Linear(in_features=input_size_audio // 2, out_features=(input_size * seq_len) // 2)
        self.normBN_diff_audio = nn.BatchNorm1d(input_size * seq_len, affine=True)

        self.speaker_fc = nn.Linear(in_features=num_speaker, out_features=input_size * seq_len)

        self.ComplementarityAttention = ComplementarityAttention(input_size=input_size, seq_len=seq_len,
                                                                 hidden_size=hidden_size,
                                                                 num_heads=num_heads,
                                                                 dropout=dropout, k=k)
        self.FusionLayerAttention = FusionLayerAttention(input_size=input_size * seq_len, dropout=dropout)
        self.Simi_Diff_Fusion_diff = Simi_Diff_Fusion(input_size=input_size * seq_len)
        self.DiffFilter = DiffFilter(seq_len=seq_len, input_size=input_size, p=p)

        self.linear = nn.Sequential(
            nn.Linear(in_features=input_size * seq_len, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout1d(p=dropout),
            nn.Linear(in_features=128, out_features=label_size)
        )

    def forward(self, vision, text_1, text_2, text_3, text_4, text_all, audio, speaker):
        simi, simi_vision, simi_text, simi_audio, mu, logvar = self.jointVAE(vision, text_all, audio)

        simi = self.normBN_simi(simi)

        diff_vision = torch.sub(vision, simi_vision)
        diff_text = torch.sub(text_all, simi_text)
        diff_audio = torch.sub(audio, simi_audio)

        r1 = self.normBNa(text_1)
        r2 = self.normBNb(text_2)
        r3 = self.normBNc(text_3)
        r4 = self.normBNd(text_4)
        text = (r1 + r2 + r3 + r4) / 4

        vision = vision.view(vision.shape[0], 2, self.input_size_img // 2)
        vision = self.fc_img(vision)
        vision = vision.view(vision.shape[0], -1)
        vision = self.normBN_img(vision)

        audio = audio.view(audio.shape[0], 2, self.input_size_audio // 2)
        audio = self.fc_audio(audio)
        audio = audio.view(audio.shape[0], -1)
        audio = self.normBN_audio(audio)

        diff_vision = diff_vision.view(diff_vision.shape[0], 2, self.input_size_img // 2)
        diff_vision = self.diff_fc_img(diff_vision)
        diff_vision = diff_vision.view(diff_vision.shape[0], -1)
        diff_vision = self.normBN_diff_img(diff_vision)

        diff_audio = diff_audio.view(diff_audio.shape[0], 2, self.input_size_audio // 2)
        diff_audio = self.diff_fc_audio(diff_audio)
        diff_audio = diff_audio.view(diff_audio.shape[0], -1)
        diff_audio = self.normBN_diff_audio(diff_audio)

        speaker = self.speaker_fc(speaker)

        diff_first = self.Simi_Diff_Fusion_diff(diff_vision, diff_text, diff_audio)
        diff = self.DiffFilter(diff_first)
        # diff = torch.cat((diff_vision, diff_text), dim=1)
        # diff = diff_vision * diff_text
        # diff = diff1 + diff2
        comple_out = self.ComplementarityAttention(vision, text, audio, diff)
        fusion_out = self.FusionLayerAttention(simi, diff, comple_out, speaker)
        if self.classer == 'parm':
            out = self.linear(fusion_out)
        else:
            x = np.array(fusion_out)
            kmeans = KMeans(n_clusters=self.output_size)
            kmeans.fit(x)
            out_label = kmeans.labels_
            out = onehotencode(out_label, self.output_size)

            # bisect_means = BisectingKMeans(n_clusters=self.output_size)
            # bisect_means.fit(x)
            # out = bisect_means.labels_
            # out = onehotencode(out_label, self.output_size)
            out = torch.tensor(out)
        return out, comple_out, simi_vision, simi_text, simi_audio, mu, logvar


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, logits, labels):
        """
        Calculates the Focal Loss.
        logits: Tensor of shape (batch_size, num_classes) - raw, unnormalized scores for each class
        labels: Tensor of shape (batch_size) - true class indices
        """
        labels_onehot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        log_p = F.log_softmax(logits, dim=-1)
        p_t = torch.exp(log_p)
        focal_loss = -self.alpha * ((1 - p_t) ** self.gamma) * log_p * labels_onehot

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.mean(torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1)))


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        cosine_similarity = F.cosine_similarity(x1, x2, dim=-1)
        return 1 - torch.mean(cosine_similarity)


class CustomLoss(nn.Module):
    def __init__(self, l1=0.5, l2=1.5, alpha=1.0, beta=50000000.0, eps=1e-8):
        super(CustomLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # self.loss_weights = torch.FloatTensor([1 / 0.46675076590376646,
        #                                        1 / 0.12209407100378447,
        #                                        1 / 0.027752748242926655,
        #                                        1 / 0.07154442241845378,
        #                                        1 / 0.1717426563344747,
        #                                        1 / 0.02640115336096594,
        #                                        1 / 0.11371418273562804]).cuda()
        #
        # self.criterion = nn.NLLLoss(self.loss_weights)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = FocalLoss()
        # self.TUBETest_s = CosineSimilarityLoss()

    def forward(self, fusion_out, comple_out, labels, labels_encoder, x_A, x_A_reconstructed, x_B, x_B_reconstructed,
                x_C, x_C_reconstructed, mu, logvar):
        labels = torch.argmax(labels, dim=1)
        tube_A = self.alpha * self.TUBETest(x_A_reconstructed, x_A)
        tube_B = self.alpha * self.TUBETest(x_B_reconstructed, x_B)
        tube_C = self.alpha * self.TUBETest(x_C_reconstructed, x_C)
        kl_loss = self.KLLoss(mu, logvar, self.beta)
        st_all = self.alpha * self.TUBETest(comple_out, labels_encoder)
        ce_loss = self.criterion(fusion_out, labels)
        comple_loss = tube_A + tube_B + tube_C + kl_loss + ce_loss + st_all
        return comple_loss

    def KLLoss(self, mu, logvar, beta):
        return -0.5 * beta * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def TUBETest(self, att, label):
        st_all = torch.tensor(0.0, dtype=torch.float32)
        for i in range(att.shape[0]):
            p = att[i]
            g = label[i]
            dot_product = torch.dot(p, g)
            p_norm = torch.norm(p)
            g_norm = torch.norm(g)
            if p_norm.item() == 0 or g_norm.item() == 0:
                cosine = torch.tensor(0)
            else:
                cosine = dot_product / (p_norm * g_norm)
            s_s = 1 - torch.square(cosine)
            if s_s.item() < 0:
                sine = torch.tensor(0)
            elif s_s.item() == 0:
                sine = torch.sqrt(s_s + self.eps)
            else:
                sine = torch.sqrt(s_s)
            if g_norm.item() == 0:
                r_all = (p_norm * cosine) / (g_norm + self.eps)
            else:
                r_all = (p_norm * cosine) / g_norm
            if r_all.item() >= 1:
                ds = self.l1 * (p_norm * sine + torch.abs(g_norm - p_norm * cosine))
            elif 0 <= r_all.item() < 1:
                ds = p_norm * sine + torch.abs(g_norm - p_norm * cosine)
            else:
                ds = self.l2 * torch.abs(p_norm * cosine - g_norm - p_norm * sine)
            if ds.item() < 0:
                raise ValueError("The program terminates when the ds value is less than 0")
            ds = -torch.log(torch.tanh(torch.reciprocal(ds)))
            st_all = torch.add(st_all, ds)
        st_all = st_all / att.shape[0]
        return st_all
