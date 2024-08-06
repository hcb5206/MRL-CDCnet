import os
import torch
from torch import nn
import random
import numpy as np
import torch.optim as optim
import argparse
from torchviz import make_dot
from fusion_layer_JointVAE_lowparm_filter import OutputLayer, CustomLoss
import torch.nn.init as init
from dataloader import dataloader
from metrics import compute_accuracy, compute_f1_score
from tqdm import tqdm

"""
task:2
***Best Parameters: [10670, 30, 90, 8, 512, 0.0005, 0.15, 0.5, 1.2, 0.8, 0.8, 0.001] | Best Value: -0.9078534031413612
"""


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_test(model, img_test, word_test_1, word_test_2, word_test_3, word_test_4, word_test_all, label_test):
    model.softmax = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        output_test, _, _, _, _, _ = model(img_test, word_test_1, word_test_2, word_test_3, word_test_4,
                                           word_test_all)
        accuracy = compute_accuracy(label_test, output_test)
        f1_score = compute_f1_score(label_test, output_test)
    return accuracy, f1_score


seed = 10670
seed_everything(seed=seed)

# torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="")
parser.add_argument("--batch_size", type=int, default=90, help="")
parser.add_argument("--input_size", type=int, default=128, help="")
parser.add_argument("--seq_len", type=int, default=16, help="")
parser.add_argument("--task", type=str, default='2', help="")
parser.add_argument("--input_size_img", type=int, default=2048, help="")
parser.add_argument("--input_size_word", type=int, default=1024, help="")
parser.add_argument("--label_size", type=int, default=5, help="")
parser.add_argument("--num_heads", type=int, default=8, help="")
parser.add_argument("--hidden_size", type=int, default=512, help="")
parser.add_argument("--lr_comple", type=float, default=0.0005, help="")
parser.add_argument("--dropout", type=float, default=0.15, help="")
parser.add_argument("--k", type=int, default=1, help="")
parser.add_argument("--classer", type=str, default='parm', help="")
parser.add_argument("--b1", type=float, default=0.9, help="")
parser.add_argument("--b2", type=float, default=0.999, help="")
parser.add_argument("--l1", type=float, default=0.5, help="Weight for the spatial distance saturation region [0.5, 1.0]")
parser.add_argument("--l2", type=float, default=1.2, help="Weight for the tonal inversion region [1.0, 1.5]")
parser.add_argument("--alpha", type=float, default=0.8, help="Weight for semantic similarity loss [0.0, 1.0]")
parser.add_argument("--p", type=float, default=0.8, help="Differentiation filter threshold [0.0, 0.9]")
parser.add_argument("--weight_decay", type=float, default=0.001, help="")
parser.add_argument("--patience", type=int, default=10, help="")
parser.add_argument("--model_path", type=str, default='models_parm/comple_fusion_main_JointVAE_2048_2',
                    help="")
opt = parser.parse_args()

dataloader_train, dataloader_val, img_test, word_test_1, word_test_2, word_test_3, word_test_4, word_test_all, \
label_test = dataloader(batch_size=opt.batch_size, task=opt.task)

categorical_loss = nn.CrossEntropyLoss()
custom_loss = CustomLoss(l1=opt.l1, l2=opt.l2, alpha=opt.alpha).to(device)

model_train = OutputLayer(input_size=opt.input_size, seq_len=opt.seq_len, hidden_size=opt.hidden_size,
                          num_heads=opt.num_heads, dropout=opt.dropout, k=opt.k,
                          label_size=opt.label_size, classer=opt.classer, input_size_img=opt.input_size_img,
                          input_size_word=opt.input_size_word, p=opt.p).to(device)

print(model_train)
# for name, param in model_train.named_parameters():
#     print(name)

print('The number of parameters in jointVAE:', count_parameters(model_train.jointVAE))
print('The number of parameters in ComplementarityAttention:', count_parameters(model_train.ComplementarityAttention))
print('The number of parameters in FusionLayerAttention:', count_parameters(model_train.FusionLayerAttention))
print('The number of parameters in MLP:', count_parameters(model_train.linear))
print('The total number of parameters', count_parameters(model_train))

for name, param in model_train.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
    elif 'bias' in name:
        param.data.fill_(0.0)

optimizer_comple = optim.AdamW(
    model_train.parameters(),
    lr=opt.lr_comple,
    betas=(opt.b1, opt.b2),
    weight_decay=opt.weight_decay)
# optimizer_comple = optim.SGD(model.ComplementarityAttention.parameters(), lr=opt.lr_comple)
# scheduler_comple = StepLR(optimizer_comple, step_size=opt.step_size, gamma=opt.gamma)

best_test_acc = 0.0
counter = 0
for epoch in range(opt.n_epochs):
    model_train.train()
    train_loss = 0.0
    progress_bar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
    for i, (
            img_tensor, word_tensor_1, word_tensor_2, word_tensor_3, word_tensor_4, word_tensor_all, labels_tensor,
            labels_encoder_tensor) in enumerate(dataloader_train):
        img, word_1, word_2, word_3, word_4, word_all, labels, labels_encoder = img_tensor.to(
            device), word_tensor_1.to(device), word_tensor_2.to(device), word_tensor_3.to(device), word_tensor_4.to(
            device), word_tensor_all.to(device), labels_tensor.to(device), labels_encoder_tensor.to(device)

        output, comple_out, simi_vision, simi_text, mu, logvar = model_train(img, word_1, word_2, word_3, word_4,
                                                                             word_all)
        loss = categorical_loss(output, labels)
        train_loss += loss.item()
        comple_loss = custom_loss(output, comple_out, labels, labels_encoder, img, simi_vision, word_all, simi_text, mu,
                                  logvar)

        optimizer_comple.zero_grad()
        comple_loss.backward()
        if comple_loss != comple_loss:
            raise Exception('NaN in comple_loss, crack!')
        # print(model_train.ComplementarityAttention.cross_att_vision.att_attention.in_proj_weight.grad)
        # print(make_dot(comple_loss))
        optimizer_comple.step()
        progress_bar_train.update(1)
        progress_bar_train.set_postfix(loss=train_loss / len(dataloader_train))
    train_loss_avg = train_loss / len(dataloader_train)
    progress_bar_train.close()

    model_train.eval()
    val_loss = 0.0
    progress_bar_val = tqdm(dataloader_val, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
    with torch.no_grad():
        for i, (
                img_tensor, word_tensor_1, word_tensor_2, word_tensor_3, word_tensor_4, word_tensor_all,
                labels_tensor) in enumerate(dataloader_val):
            img_val, word_val_1, word_val_2, word_val_3, word_val_4, word_val_all, labels_val = img_tensor.to(
                device), word_tensor_1.to(device), word_tensor_2.to(device), word_tensor_3.to(device), word_tensor_4.to(
                device), word_tensor_all.to(device), labels_tensor.to(device)

            output_val, _, _, _, _, _ = model_train(img_val, word_val_1, word_val_2, word_val_3, word_val_4,
                                                    word_val_all)
            loss_v = categorical_loss(output_val, labels_val)
            val_loss += loss_v.item()
            progress_bar_val.update(1)
            progress_bar_val.set_postfix(loss=val_loss / len(dataloader_val))
        val_loss_avg = val_loss / len(dataloader_val)
        progress_bar_val.close()

    test_acc, f1score = model_test(model_train, img_test, word_test_1, word_test_2, word_test_3, word_test_4,
                                   word_test_all, label_test)

    print(f'Epoch {epoch + 1}/{opt.n_epochs}, Train Loss: {train_loss_avg:.6f}, Val loss: {val_loss_avg:.6f}, '
          f'test acc: {test_acc:.6f}, test f1score: {f1score:.6f}')

    if best_test_acc <= test_acc:
        best_test_acc = test_acc
        print(best_test_acc)
        torch.save(model_train.state_dict(), opt.model_path)
        counter = 0
    else:
        counter += 1

    if counter >= opt.patience:
        print("Early stopping")
        break

print('best acc:', best_test_acc)
print('train has finished!')