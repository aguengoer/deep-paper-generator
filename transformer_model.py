import torch


# Create the Transformer model
class TransformerModel(torch.nn.Module):
    def __init__(self, d_model, num_layers, num_heads, hidden_dim, dropout, src_input_dim, trg_input_dim):
        super().__init__()

        # self.embedding = torch.nn.Embedding(input_dim, d_model)
        self.src_embedding = torch.nn.Embedding(src_input_dim, d_model)
        self.trg_embedding = torch.nn.Embedding(trg_input_dim, d_model)
        self.transformer = torch.nn.Transformer(d_model, num_heads, num_layers, dim_feedforward=hidden_dim,
                                                dropout=dropout)
        self.fc = torch.nn.Linear(d_model, trg_input_dim)

    def forward(self, src, trg):
        # src shape: (src_len, batch_size)
        # trg shape: (trg_len, batch_size)

        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        # Pass the inputs through the transformer model
        output = self.transformer(src, trg)

        # Pass the output through a linear layer to get the final prediction
        output = self.fc(output)

        return output

