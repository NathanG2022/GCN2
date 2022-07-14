import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation
from torch.autograd import Variable
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from dgl.nn import EdgeWeightNorm, GraphConv
import dgl.function as fn
from dgl.geometry.pytorch import FarthestPointSampler

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

def build_karate_club_graph():
    '''
    All 78 edges are stored in two numpy arrays, one for the source endpoint and the other for the target endpoint
    '''
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    #Edges are directional in DGL; make them bidirectional
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    #Building diagram
    return dgl.DGLGraph((u, v))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    G = build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    # Visualize the graph by converting it into a networkx graph
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.savefig('graph.png')

    embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
    G.ndata['feat'] = embed.weight
    print(G.ndata['feat'][2])
    print(G.ndata['feat'][[10, 11]])

    #Initialize network instance
    net = GCN(5, 5, 2)
    print('net:', net)

    inputs = embed.weight
    labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1])  # their labels are different

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
    all_logits = []
    for epoch in range(50):
        logits = net(G, inputs)
        # we save the logits for visualization later
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    draw(0)
    plt.savefig('0.png')

    ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    plt.savefig('1.png')