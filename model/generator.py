from torch import nn

from model.layers.basic_module import MLP
from model.layers.global_graph import SelfAttentionWithKeyFCLayer
import torch


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, steps):
        super().__init__()
        self.mlp = MLP(
            in_channel=hidden_size, out_channel=hidden_size * steps
        )
        self.embedding = nn.Embedding(n_classes, hidden_size)
        self.dense = MLP(
            in_channel=input_size, out_channel=hidden_size, activation="leaky"
        )
        self.dense2 = MLP(
            in_channel=input_size, out_channel=hidden_size, activation="leaky"
        )
        self.fc = MLP(in_channel=hidden_size, out_channel=2, activation="leaky")
        self.hidden_size = hidden_size
        # self.self_att = SelfAttentionWithKeyFCLayer(
        #     in_channels=hidden_size, 
        #     global_graph_width=hidden_size
        # )

    def forward(self, z, labels):
        # bs = h.shape[0]
        bs = z.shape[0]
        x1 = self.dense(z) + self.embedding(labels)
        output = self.mlp(x1)
        output = output.view(bs, -1, self.hidden_size)
        # dense_h = self.dense2(h)
        # output = self.self_att(output,dense_h, valid_lens)
        
        output = self.fc(output)
        return output


class GeneratorV2(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, steps):
        super().__init__()
        self.mlp = MLP(
            in_channel=hidden_size, out_channel=hidden_size * steps, activation="leaky"
        )
        self.embedding = nn.Embedding(n_classes, hidden_size)
        self.dense = MLP(
            in_channel=input_size, out_channel=hidden_size, activation="leaky"
        )
        self.dense2 = MLP(
            in_channel=input_size, out_channel=hidden_size, activation="leaky"
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)
        self.hidden_size = hidden_size

        self.self_att = nn.ModuleList(
            [
                SelfAttentionWithKeyFCLayer(
                    in_channels=hidden_size, 
                    global_graph_width=hidden_size
                )

                for _ in range(10)
            ]
        )


        # self.gru = nn.GRU(
        #     input_size=hidden_size,
        #     hidden_size=hidden_size, 
        #     batch_first=True,
        #     bidirectional=True,
        #     num_layers=10
        # )

        # self.dense3 = nn.Linear(
        #     in_features=2*hidden_size,
        #     out_features=hidden_size
        # )

        self.position_encoding = nn.Parameter(
            torch.randn(1, 50 , 2)
        )
    def forward(self, z, labels, h, valid_lens):
        bs = z.shape[0]
        x1 = self.dense(z) + self.embedding(labels)
        output = self.mlp(x1)
        output = output.view(bs, -1, self.hidden_size)
        dense_h = self.dense2(h)
        for self_att in self.self_att:
            output = self_att(output, dense_h, valid_lens)
        # output,_ = self.gru(output)
        # output = self.dense3(output)
        output = self.fc(output) + self.position_encoding
        return output


if __name__ == "__main__":
    input_size = 12
    hidden_size = 100
    steps = 50
    n_class = 8
    z = torch.randn(64, input_size)
    labels = torch.randint(low=0, high=8, size=[64])
    h = torch.randn(64, hidden_size)
    generator = Generator(
        input_size, hidden_size, n_class, steps
    )
    print(
        generator(z, labels, h)
    )

