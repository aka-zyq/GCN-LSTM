import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Propogator(nn.Module):
    def __init__(self, node_dim):    
        super(Propogator, self).__init__()
        self.node_dim = node_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), 
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),  
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), 
            nn.Tanh()
        )

    def forward(self, node_representation, adjmatrixs):
        a = torch.bmm(adjmatrixs, node_representation)
        joined_input1 = torch.cat((a, node_representation), 2)
        z = self.update_gate(joined_input1)
        r = self.reset_gate(joined_input1)
        joined_input2 = torch.cat((a, r * node_representation), 2)   
        h_hat = self.tansform(joined_input2)      
        output = (1 - z) * node_representation + z * h_hat
        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)   
        self.state_dim = 256
        self.n_steps = 5
        self.propogator = Propogator(self.state_dim)
        self.out1 = nn.Sequential(
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Tanh()
        )
        self.out2 = nn.Sequential(    # this is new adding for graph-level outputs
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Sigmoid()
        )
        self._initialization()
    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, images, adjmatrixs, lengths): 
        lengths = torch.Tensor(lengths).reshape(-1, 1).to(device)
        embeddings = self.embed(images).to(device)
        node_representation = embeddings
        init_node_representation = node_representation
        for i_step in range(self.n_steps):    # time_step updating
            node_representation = self.propogator(node_representation, adjmatrixs)   
        gate_inputs = torch.cat((node_representation, init_node_representation), 2)
        gate_outputs = self.out1(gate_inputs)
        features = torch.sum(gate_outputs, 1)    
        features = features / lengths

        return features


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(embed_size, vocab_size)
        self.linear2 = nn.Linear(vocab_size, vocab_size)
        self.linear3 = nn.Linear(vocab_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # <class 'torch.nn.utils.rnn.PackedSequence'>
        outputs = self.linear1(packed)
        outputs = self.linear2(outputs)
        outputs = self.linear3(outputs)
        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)   # torch.Size([512, 1, 256])  [batch_size, time_steps, embeddingVector_size]
        for i in range(self.max_seg_length):                     # max_seg_length = 20
            outputs = self.linear1(inputs)
            outputs = self.linear2(outputs)
            outputs = self.linear3(outputs)
            _, predicted = outputs.max(1)                        # predicted: (batch_size) is the position corresponding the max
            sampled_ids.append(predicted)   # the unit of predicted arrays is the label of the vocab_size 
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
