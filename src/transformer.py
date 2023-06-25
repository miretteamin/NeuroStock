import torch
import torch.nn as nn
import numpy as np

# [2600, 10, 5] [companies*batch, days, features]
seq_len = 15

d_k = 12
d_v = 12
n_heads = 1
ff_dim = 12
input_shape = (2600, 15, 7)
linear_time_feature= 1 #4

class Time2Vector(nn.Module):
    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = 15
        self.weights_linear = nn.Parameter(torch.rand(self.seq_len))
        self.bias_linear = nn.Parameter(torch.rand(self.seq_len))
        self.weights_periodic = nn.Parameter(torch.rand(self.seq_len))
        self.bias_periodic = nn.Parameter(torch.rand(self.seq_len))

    def forward(self, x):
        #print("time2Vector embedding input {}".format(np.shape(x)))
        x = torch.mean(x[:,:,:4], dim=-1)
        #print("x input {}".format(np.shape(x)))
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = time_linear.unsqueeze(-1)
        #print(" time_linear {}".format(np.shape(time_linear)))

        time_periodic = torch.sin(x * self.weights_periodic + self.bias_periodic)
        time_periodic = time_periodic.unsqueeze(-1)
        #print(" time_periodic {}".format(np.shape(time_periodic)))
        #print("time2Vector embedding output {}".format(np.shape(torch.cat([time_linear, time_periodic], dim=-1))))
        P = torch.cat([time_linear, time_periodic], dim=-1)
        #print("P {}".format(np.shape(P)))
        return torch.cat([time_linear, time_periodic], dim=-1)
    
# time2Vector embedding input torch.Size([2480, 15, 5])
# x input torch.Size([2480, 15])
#  time_linear torch.Size([2480, 15, 1])
#  time_periodic torch.Size([2480, 15, 1])
# P torch.Size([2480, 15, 2])
    
class SingleAttention(nn.Module):
    def __init__(self, d_k, d_v, num_timeseries_features):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_timeseries_features = num_timeseries_features
        self.query = nn.Linear(in_features=self.num_timeseries_features, out_features=self.d_k)
        self.key = nn.Linear(in_features=self.num_timeseries_features, out_features=self.d_k)
        self.value = nn.Linear(in_features=self.num_timeseries_features, out_features=self.d_v)

    def forward(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        attn_weights = torch.matmul(q, k)
        attn_weights = attn_weights / np.sqrt(self.d_k)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        v = self.value(inputs[2])
        attn_out = torch.matmul(attn_weights, v)
        return attn_out
    
class MultiAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, num_timeseries_features):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_timeseries_features = num_timeseries_features
        self.attn_heads = nn.ModuleList([SingleAttention(d_k, d_v, num_timeseries_features= self.num_timeseries_features) for _ in range(n_heads)])
        self.linear = nn.Linear(in_features=d_v*n_heads, out_features=self.num_timeseries_features)

    def forward(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = torch.cat(attn, dim=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear


class TransformerEncoder(nn.Module):
    def __init__(self, d_k, d_v, n_heads, ff_dim, num_timeseries_features, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.num_timeseries_features = num_timeseries_features
        self.attn_multi = MultiAttention(d_k, d_v, n_heads, num_timeseries_features= self.num_timeseries_features)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_normalize = nn.LayerNorm(normalized_shape=self.num_timeseries_features)

        self.ff_conv1D_1 = nn.Conv1d(in_channels=input_shape[-2], out_channels=ff_dim, kernel_size=1)
        self.ff_conv1D_2 = nn.Conv1d(in_channels=ff_dim, out_channels=15, kernel_size=1)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_normalize = nn.LayerNorm(normalized_shape=(d_v*n_heads,))

    def forward(self, inputs):
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        # print(np.shape(attn_layer))
        # print(np.shape(inputs[0]))

        attn_layer = self.attn_normalize(inputs[0] + attn_layer)
        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        #ff_layer = self.ff_normalize(inputs[0].transpose(1,0) + ff_layer)
        return ff_layer



class HistoricalTransformer(nn.Module):
    def __init__(self, hidden_size, d_k, d_v, n_heads, ff_dim, num_timeseries_features, output_size=64):
        super(HistoricalTransformer, self).__init__()
        self.num_timeseries_features= num_timeseries_features+2
        self.hidden_size = hidden_size
        self.time_embedding = Time2Vector(hidden_size)
        self.attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, num_timeseries_features=self.num_timeseries_features)
        self.attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, num_timeseries_features= self.num_timeseries_features)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_v, output_size)


    def forward(self, x):
        # [2600, 10, 5] [companies*batch, days, features]
        time_feat = self.time_embedding(x)
        x = torch.cat([x, time_feat], dim=-1)
        x = self.attn_layer1((x,x,x))
        x = self.attn_layer2((x,x,x))
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        out = self.fc1(x[:, -1, :]) #(n_companies*batch_size, hidden_size)
        # x = nn.ReLU()(x) (n_companies*batch_size, seq_len, embeddings)

        return out

#model = HistoricalTransformer(hidden_size=32, d_k=64, d_v=64, n_heads=8, ff_dim=256)
#print(model)