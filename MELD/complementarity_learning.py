import torch
import torch.nn as nn
import torch.nn.functional as F


class Simi_Diff_Fusion(nn.Module):
    def __init__(self, input_size, seq_len):
        super(Simi_Diff_Fusion, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.wc = nn.Linear(in_features=input_size * seq_len, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xc = self.softmax(self.wc(x)).permute(0, 2, 1)
        out = torch.matmul(xc, x).squeeze()
        output = self.tanh(out)
        output = output.view(output.shape[0], self.seq_len, self.input_size)
        return output


class DiffAttention(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, dropout, k):
        super(DiffAttention, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.k = k
        self.wk = Simi_Diff_Fusion(input_size=input_size, seq_len=seq_len)
        self.wo = nn.Linear(in_features=input_size, out_features=seq_len)
        self.wc = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout_s = nn.Dropout1d(p=dropout)

    def forward(self, xk, diff):
        if self.k == 1:
            x = xk
        else:
            x = self.wk(xk)
        x = self.wo(x)
        out = self.wc(torch.matmul(x, diff))
        out = out.permute(0, 2, 1)
        out = self.bn(out).permute(0, 2, 1)
        out = self.tanh(out)
        out = self.dropout_s(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_heads, dropout, k):
        super(CrossAttention, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.k = k
        self.att_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm_att = nn.LayerNorm(input_size)
        self.lc_att = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.Q = DiffAttention(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size, dropout=dropout, k=k)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm_cross = nn.LayerNorm(hidden_size)

    def forward(self, x, xk, diff):
        x = x.permute(1, 0, 2)
        x_att, _ = self.att_attention(x, x, x)
        x_att = x_att.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x_att = self.layer_norm_att(x + x_att)
        x_att = self.lc_att(x_att)
        x_att = x_att.permute(1, 0, 2)
        x_q = self.Q(xk, diff)
        x_q = x_q.permute(1, 0, 2)
        out, _ = self.cross_attention(x_q, x_att, x_att)
        x_q = x_q.permute(1, 0, 2)
        out = out.permute(1, 0, 2)
        out = self.layer_norm_cross(x_q + out)
        return out


class OutputAttention(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, dropout):
        super(OutputAttention, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.wt = nn.Linear(in_features=input_size, out_features=1)
        self.wm = nn.Linear(in_features=hidden_size, out_features=1)
        self.wo = nn.Linear(in_features=seq_len, out_features=input_size * seq_len)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(num_features=input_size * seq_len)
        self.dropout_s = nn.Dropout1d(p=dropout)

    def forward(self, x, diff):
        xi = self.wm(x).squeeze()
        w3 = self.tanh(self.wt(diff))
        att = self.softmax(torch.matmul(xi, w3)).permute(0, 2, 1)
        pi = torch.matmul(att, xi)
        out = self.wo(pi).squeeze()
        out = self.bn(out)
        out = self.tanh(out)
        out = self.dropout_s(out)
        return out


class ComplementarityAttention(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_heads, dropout, k):
        super(ComplementarityAttention, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.k = k
        self.cross_att_vision = CrossAttention(input_size, seq_len, hidden_size, num_heads, dropout, k)
        self.cross_att_text = CrossAttention(input_size, seq_len, hidden_size, num_heads, dropout, k)
        self.cross_att_audio = CrossAttention(input_size, seq_len, hidden_size, num_heads, dropout, k)
        self.output_attention = OutputAttention(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size,
                                                dropout=dropout)

    def forward(self, vision, text, audio, diff):
        vision_sk = torch.cat((text.unsqueeze(1), audio.unsqueeze(1)), dim=1)
        text_sk = torch.cat((vision.unsqueeze(1), audio.unsqueeze(1)), dim=1)
        audio_sk = torch.cat((vision.unsqueeze(1), text.unsqueeze(1)), dim=1)

        vision = vision.view(vision.shape[0], self.seq_len, self.input_size)
        text = text.view(text.shape[0], self.seq_len, self.input_size)
        audio = audio.view(audio.shape[0], self.seq_len, self.input_size)
        diff = diff.view(diff.shape[0], self.seq_len, self.input_size)
        att_vision = self.cross_att_vision(vision, vision_sk, diff).unsqueeze(1)
        att_text = self.cross_att_text(text, text_sk, diff).unsqueeze(1)
        att_audio = self.cross_att_audio(audio, audio_sk, diff).unsqueeze(1)
        att_in = torch.cat((att_vision, att_text, att_audio), dim=1)
        output_att = self.output_attention(att_in, diff)
        return output_att


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
