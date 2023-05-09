"""
This is an implementation of ProNet model

"""

from torch_geometric.nn import inits, MessagePassing
from torch_geometric.nn import radius_graph
from torch_geometric.nn import GINConv, TopKPooling, GCNConv, BatchNorm, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import random
from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter
from torch_sparse import matmul

import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

import numpy as np
import math

num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6


def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
        A layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """

    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class GAT_4layer(torch.nn.Module):
    def __init__(self, num_feature, num_gene, nhid=256):
        super(GAT_4layer, self).__init__()
        self.conv1 = GATConv(num_feature, nhid, num_heads=4)
        self.bn1 = BatchNorm(nhid)
        self.conv2 = GATConv(nhid, nhid, num_heads=4)
        self.bn2 = BatchNorm(nhid)
        self.conv3 = GATConv(nhid, nhid, num_heads=4)
        self.bn3 = BatchNorm(nhid)
        self.conv4 = GATConv(nhid, nhid, num_heads=4)
        self.bn4 = BatchNorm(nhid)

        self.lin1 = torch.nn.Linear(nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid // 2)
        self.lin3 = torch.nn.Linear(nhid // 2, num_gene)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()
        self.bn4.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        # print(x.shape)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # print(x1.shape)

        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x4 = F.relu(self.bn4(self.conv4(x3, edge_index)))
        # x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # print(x.shape)
        # print(x4.shape)

        x = x1 + x2 + x3 + x4
        # print(x.shape)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        y = torch.squeeze(self.lin3(x), dim=1)

        return y

class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            output_channels,
            num_radial,
            num_spherical,
            num_layers,
            mid_emb,
            act=swish,
            num_pos_emb=16,
            dropout=0,
            level='allatom'
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        # print(hidden_channels) = 20

        # self.conv0 = GAT_4layer(hidden_channels, hidden_channels, nhid=20)
        # self.conv1 = GAT_4layer(hidden_channels, hidden_channels, nhid=20)
        # self.conv2 = GAT_4layer(hidden_channels, hidden_channels, nhid=20)
        self.conv3 = GAT_4layer(hidden_channels, hidden_channels, nhid=20)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        if level == 'aminoacid':
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == 'backbone' or level == 'allatom':
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)

        self.SelfAttention0 = SelfAttention(num_attention_heads=4, input_size=20, hidden_size=20, hidden_dropout_prob=0.2)
        self.SelfAttention1 = SelfAttention(num_attention_heads=4, input_size=20, hidden_size=20, hidden_dropout_prob=0.2)
        self.SelfAttention2 = SelfAttention(num_attention_heads=4, input_size=20, hidden_size=20, hidden_dropout_prob=0.2)
        self.SelfAttention3 = SelfAttention(num_attention_heads=4, input_size=20, hidden_size=20, hidden_dropout_prob=0.2)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(4 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.SelfAttention0.reset_parameters()
        self.SelfAttention1.reset_parameters()
        self.SelfAttention2.reset_parameters()
        self.SelfAttention3.reset_parameters()


        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))

        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)

        # h0 = self.conv0(x_lin_1, edge_index, batch)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        # h1 = self.conv1(x_lin_1, edge_index, batch)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        # h2 = self.conv2(x_lin_1, edge_index, batch)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)
        # print(x_lin_1)
        # print(x_lin_1.shape)
        # print(x_lin_2.shape)
        # print(edge_index.shape)
        # print(feature0.shape)
        # print(feature1.shape)
        # print(feature2.shape)

        h3 = self.conv3(x_lin_1, edge_index, batch)
        # print(h3.shape)
        h3 = self.lin3(h3)
        h3 = self.act(h3)
        h3 = self.dropout(h3)


        # print(h0.shape)
        # print(h1.shape)
        # print(h3.shape)
        h0 = self.SelfAttention0(h0)
        h1 = self.SelfAttention1(h1)
        h2 = self.SelfAttention2(h2)
        h3 = self.SelfAttention3(h3)

        h = torch.cat((h0, h1, h2, h3), 1)
        for lin in self.lins_cat:
            h = self.act(lin(h))

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h))
        h = self.final(h)
        return h


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

    # def reset_parameters(self):



class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention ""heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        # self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.attn_dropout = nn.Dropout(0.2)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print(x)
        # print(x.shape)
        return x.permute(0, 2, 1)

    def reset_parameters(self):
        self.query.reset_parameters()
        self.dense.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        # print(mixed_query_layer)
        # print(mixed_query_layer.shape)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # print(hidden_states.shape)
        # print(input_tensor.shape)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ProNet(nn.Module):
    r"""
         The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.

        Args:
            level: (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
            num_blocks (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            mid_emb (int, optional): Embedding size used for geometric features. (default: :obj:`64`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
            max_num_neighbors (int, optional): Max number of neighbors during graph construction. (default: :obj:`32`)
            int_emb_layers (int, optional): Number of embedding layers in the interaction block. (default: :obj:`3`)
            out_layers (int, optional): Number of layers for features after interaction blocks. (default: :obj:`2`)
            num_pos_emb (int, optional): Number of positional embeddings. (default: :obj:`16`)
            dropout (float, optional): Dropout. (default: :obj:`0`)
            data_augment_eachlayer (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
            euler_noise (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)

    """

    def __init__(
            self,
            level='aminoacid',
            num_blocks=4,
            hidden_channels=128,
            out_channels=1,
            mid_emb=64,
            num_radial=6,
            num_spherical=2,
            cutoff=10.0,
            max_num_neighbors=32,
            int_emb_layers=3,
            out_layers=2,
            num_pos_emb=16,
            dropout=0,
            data_augment_eachlayer=True,
            euler_noise=True,
            Train = True,
            epoch = 1,
            num_attention_heads = 4,
            input_size = 20,
            hidden_size = 20,
            hidden_dropout_prob = 0.2,
    ):
        super(ProNet, self).__init__()
        self.Train = Train
        self.epoch = epoch
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        if level == 'aminoacid':
            self.embedding = Embedding(num_aa_type, hidden_channels)
        elif level == 'backbone':
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs, hidden_channels)
        elif level == 'allatom':
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels)
        else:
            print('No supported model!')

        self.SelfAttention = SelfAttention(num_attention_heads=num_attention_heads, input_size=input_size, hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob)
        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=out_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level
                )
                for _ in range(num_blocks)
            ]
        )


        self.lins_out = torch.nn.ModuleList()
        for _ in range(out_layers - 1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data):

        # z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        # print(batch_data)
        # print(batch_data.shape) DataBatch(side_chain_embs=[443, 8], bb_embs=[443, 6], x=[443, 1], coords_ca=[443, 3], coords_n=[443, 3], coords_c=[443, 3], id=[1], batch=[443], ptr=[2])
        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        # print("shape of z\n")
        # print(z.shape)

        # print(batch)
        # print(batch.shape)
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        # print("shape of bbmbs \n")
        # print(bb_embs)

        device = z.device
        if self.Train == True:
            seqs_zeros = torch.zeros(size=z.shape, dtype=torch.long)
            seqs_rand = torch.rand(size=z.shape)
            mask_percentage = 1 - random.random()/math.log(math.exp(self.epoch)+1)
            z = torch.where(seqs_rand > mask_percentage, z, seqs_zeros)
        else:
            seqs_zeros = torch.zeros(size=z.shape, dtype=torch.long)
            seqs_rand = torch.rand(size=z.shape)
            mask_percentage = 1
            z = torch.where(seqs_rand > mask_percentage, z, seqs_zeros)


        if self.level == 'aminoacid':
            x = self.embedding(z)
            # self.embedding = Embedding(num_aa_type, hidden_channels)
            # mask_percentage = 1 - random.random() / (8 + 2 * math.sin(self.epoch))
            # num_aa_type = 26
            # hidden_channels = 128
        elif self.level == 'backbone':
            x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs], dim=1)
            # print("shape of x:\n")
            # print(x.shape)
            x = self.embedding(x)
            # print("shape of x:\n")
            # print(x.shape)
        elif self.level == 'allatom':
            x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs, side_chain_embs],
                          dim=1)
            x = self.embedding(x)
        else:
            print('No supported model!')

        # print(type(batch))
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)

        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i - 1) % num_nodes
        refi1 = (i + 1) % num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)

        feature0 = self.feature0(dist, theta, phi)

        if self.level == 'backbone' or self.level == 'allatom':
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7

            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)

            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1) / Or1_z_length,
                                 (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1) / Or2_z_length,
                                 (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3, len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat(
                (self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)), 1)

        elif self.level == 'aminoacid':
            refi = (i - 1) % num_nodes

            refj0 = (j - 1) % num_nodes
            refj = (j - 1) % num_nodes
            refj1 = (j + 1) % num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1)
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        # print(x.shape)
        x = self.SelfAttention(x)

        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                # add gaussian noise to features
                gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                x += gaussian_noise


            # x_mask

            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)
            # print(x.shape)
            # print(x) -- tensor([[ 0.0256,  0.0150,  0.0236,  ..., -0.0274,  0.0199, -0.0287],
            #         [-0.0257,  0.0048,  0.0516,  ..., -0.0051,  0.0336, -0.0321],
            #         [ 0.0425,  0.0407,  0.0364,  ..., -0.0022, -0.0336, -0.0355],
            #         ...,
            #         [-0.0615,  0.0184, -0.0192,  ...,  0.0028, -0.0020, -0.0353],
            #         [-0.0574,  0.0303, -0.0184,  ..., -0.0007, -0.0070, -0.0368],
            #         [-0.0650,  0.0215, -0.0188,  ...,  0.0027, -0.0019, -0.0318]],
            #        grad_fn=<AddmmBackward0>)
            # print(type(x)) --  <class 'torch.Tensor'>
            # print(x.shape) # -- torch.Size([10081, 128])
        ##20230223
        # print(x.shape)
        # torch.Size([7997, 128]) / torch.Size([11000, 128]) its will change with different batch
        # print(x.shape)


        # ZY -- 2023/03/06
        # lin1 = torch.nn.Linear(128, 20)
        # pred_y = lin1(x)  #[10081, 20]
        # print(pred_y)
        # print(x.shape)
        #
        # print(batch.shape)
        # print(batch)

        # y = scatter(x, batch, dim=0)
        # torch.Size([32, 128])

        # for lin in self.lins_out:
        #      y = self.relu(lin(y)) # torch.Size([32, 128])
        #      y = self.dropout(y) # torch.Size([32, 128])
        # y = self.lin_out(y) # torch.Size([32, 1])
        # print(y)
        # tensor([[-0.0787],[-0.1091],...,grad_fn=<AddmmBackward0>)
        # print("shape of x:\n")
        # print(x.shape)
        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
"""
else :
    seqs_zeros = torch.zeros(size=z.shape, dtype=torch.long)
    seqs_rand = torch.rand(size=z.shape)
    mask_percentage = 1
    z = torch.where(seqs_rand > mask_percentage, z, seqs_zeros)
"""