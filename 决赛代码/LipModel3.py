import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.3

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        self.shorcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shorcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shorcut(x)
        out = self.relu(out)
        return out


class LipModel(nn.Module):
    def __init__(self, c_in, num_class):
        super(LipModel, self).__init__()
        self.show_log = False

        # 3D卷积
        self.conv3d1 = nn.Conv3d(c_in, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 3), stride=(1, 2, 3), padding=(0, 0, 0))

        # ResNet
        self.res_channel = 64
        self.resnet = nn.Sequential(
            # 3 4 6 3
            self.ResLayer(64, n_block=1, stride=1),
            self.ResLayer(128, n_block=2, stride=2),
            self.ResLayer(256, n_block=2, stride=2),
            self.ResLayer(512, n_block=2, stride=2)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        hidden_size = 256
        self.res_linear = nn.Linear(self.res_channel, hidden_size)
        # self.linear_bn = nn.BatchNorm1d(hidden_size)  # (bsz, d, 256)
        self.linear_bn = nn.Dropout(dropout)

        # Attention
        self.att_w = nn.Linear(hidden_size, 3, bias=False)
        self.attn_bn = nn.BatchNorm1d(hidden_size)

        # LSTM
        self.lstm = nn.LSTM(256, 256, num_layers=2, dropout=0.3, bidirectional=True)
        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers=1, bidirectional=True)

        # Classfication
        self.classfication = nn.Linear(2*hidden_size, num_class)
        self.classfication10 = nn.Linear(hidden_size, 10)


    def ResLayer(self, out_channel, n_block, stride):
        strides = [stride] + [1] * (n_block-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.res_channel, out_channel, stride))
            self.res_channel = out_channel
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)


    def forward(self, inputs, targets=None):
        '''

        :param inputs: shape:(bsz, channel, timestep, height, width)
        :param targets: tensor向量
        :return:
        '''
        # Conv3D
        out = self.conv3d1(inputs)
        if self.show_log: print('conv3d_1:',out.size())
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool3d1(out)
        if self.show_log: print('pool3d_1:', out.size())

        # ResNet
        out = torch.transpose(out, 1, 2).contiguous()
        bsz, d, c, h, w = out.size()
        out = out.view(-1, c, h, w)
        if self.show_log: print('resize:', out.size())
        out = self.resnet(out)
        if self.show_log: print('resnet:', out.size())
        out = self.avg_pool(out)
        if self.show_log: print('avg_pool:', out.size())
        out = out.view(bsz*d, self.res_channel)
        if self.show_log: print('resize:', out.size())
        out = self.res_linear(out)
        if self.show_log: print('res_linear:', out.size())
        out = self.linear_bn(out)
        out = out.view(bsz, d, -1)
        if self.show_log: print('linear_bn:', out.size())  # (bsz, d, 256)

        # RNN
        rnn_out = out.transpose(0, 1).contiguous()
        rnn_out = self.lstm(rnn_out)[0]
        rnn_out = rnn_out.transpose(0, 1)
        if self.show_log: print('lstm:', rnn_out.size())  #(bsz, time_step, 2*hidden_size)

        # Attention
        att_out = self.att_w(out)  #(bsz, d, 3)
        att_out = att_out.transpose(1, 2)  #(bsz, 3, d)
        att_out = F.softmax(att_out, -1)  #(bsz, 3, d)
        att_out = torch.einsum('bdh,bid->bih', out, att_out)  #(bsz, 3, 256)
        if self.show_log: print('Attention:', att_out.size())
        att_out = att_out.contiguous().view(-1, att_out.size(-1))  #(bsz*3, 256)
        self.attn_bn(att_out)
        if self.show_log: print('Attention BN:', att_out.size())

        # classfication1
        rnn_out = self.classfication(rnn_out)
        logit = F.softmax(rnn_out, -1)
        logit = torch.sum(logit, dim=1)
        result = (logit,)

        # classfication2
        logit10 = self.classfication10(att_out)  # (bsz*3, 10)


        # logit融合
        # logit_help = F.softmax(logit10, dim=-1).view(bsz, 3, 10)
        # prob1, prob2, prob3 = logit_help.chunk(3, 1)
        # prob1, prob2, prob3 = prob1.view(bsz, 10, 1), prob2.view(bsz, 10, 1), prob3.view(bsz, 10, 1)
        # prob_help = torch.einsum('bgi,bsi->bgs', prob1, prob2) # (bsz, 10, 10)
        # prob_help = prob_help.contiguous().view(bsz, 100, 1)
        # prob_help = torch.einsum('bgi,bsi->bgs', prob_help, prob3) # (bsz, 100, 10)
        # prob_help = prob_help.contiguous().view(bsz, 1000)

        # logit = logit + prob_help
        # result = (logit,)


        # loss
        if torch.is_tensor(targets):
            labels_per_int = get_labels_per_int(targets)
            logit10_log = -F.log_softmax(logit10, -1).view(-1, 10)
            labels_per_int = labels_per_int.view(-1)
            loss2 = logit10_log.gather(dim=-1, index=labels_per_int[:, None]).squeeze()

            log_sm = torch.mean(-F.log_softmax(rnn_out, -1), dim=1)
            loss1 = log_sm.gather(dim=-1, index=targets[:, None]).squeeze()

            loss = loss1.mean() + loss2.mean()
            # loss = loss1
            result = (logit, loss)

        return result

def get_labels_per_int(x):
    x_cp = x.clone()
    labels_per_int = []
    for i in range(3):
        labels_per_int.append(x_cp % 10)
        x_cp /= 10
    labels_per_int = torch.stack(labels_per_int, 0).T
    labels_per_int = torch.index_select(labels_per_int, dim=-1, index=torch.tensor([2, 1, 0], device=x.device))
    return labels_per_int


if __name__ == '__main__':
    lip_net = LipModel(c_in=3, num_class=1000)
    a = torch.randn(5, 3, 11, 120, 180)
    labels = torch.tensor(np.random.randint(0, 1000, (5,)))
    out = lip_net(a, labels)
    print(out[0].size())

    # a = torch.tensor([1,254,62,25,125], device='cuda')
    # b = get_labels_per_int(a)
    # print(a, b)
