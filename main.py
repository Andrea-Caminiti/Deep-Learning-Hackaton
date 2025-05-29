import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
import random
import numpy as np
from source.loadData import GraphDataset, BatchedGraphDataset
import os
import pandas as pd
from source.loss import NoisyCrossEntropyLoss
from tqdm import tqdm 
from torch_geometric.utils import degree
import gc

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data

class G2SAGEConv(MessagePassing):
    def __init__(self, in_channels, edge_feat, p=1.0):
        super(G2SAGEConv, self).__init__(aggr='mean')
        self.lin = nn.Linear(edge_feat, edge_feat)
        self.gating_linear = nn.Linear(in_channels, edge_feat)
        self.p = p

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin(out)
        out = F.relu(out)
        
        tau_hat = torch.sigmoid(self.gating_linear(x))  # (N, F_out)
        
        row, col = edge_index
        tau_diff = torch.abs(tau_hat[row] - tau_hat[col]) ** self.p  # (E, F_out)

        tau_diff = tau_diff * edge_attr  # (E, F_out)

        
        tau_sum = torch.zeros_like(tau_hat)
        tau_sum = tau_sum.index_add(0, row, tau_diff)

        tau = torch.tanh(tau_sum)

        updated_x = (1 - tau) * x + tau * out
        return updated_x

    def message(self, x_j, edge_attr):
        
        return edge_attr * x_j  
    
class DeepG2SAGEGNN(nn.Module):
    def __init__(self, in_features, num_layers, edge_feat):
        super(DeepG2SAGEGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(G2SAGEConv(in_features, edge_feat))
        for _ in range(num_layers - 1):
            self.layers.append(G2SAGEConv(edge_feat, edge_feat))
        self.classifier = nn.Linear(edge_feat, 6)

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            identity = x
            x = layer(x, edge_index, edge_attr)
            x += identity  # residual

        # Pool node features to get graph-level embeddings
        graph_embeddings = global_mean_pool(x, batch)  # (num_graphs, hidden_features)

        # Classify graphs
        out = self.classifier(graph_embeddings)
        return out
    
def train(data_loader):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        x = deg.view(-1, 1)
        optimizer.zero_grad()
        output = model(x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(data_loader, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)    
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
            x = deg.view(-1, 1)
            output = model(x, data.edge_index, data.edge_attr, data.batch)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    global model, optimizer, criterion, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters for the GCN model
    input_dim = 1
    edge_feat = 7
    num_layers = 5

    # Initialize the model, optimizer, and loss criterion
    model = DeepG2SAGEGNN(input_dim, num_layers, edge_feat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = NoisyCrossEntropyLoss(0.2)

    # Prepare test dataset and loader
    if args.jsonl:
        test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]

        if args.train_path:
            num_epochs = 150 
            for epoch in tqdm(range(num_epochs), desc=f'Training...', leave=False):            
                with open(args.train_path, 'r') as f:
                    length = len(f.readlines())
                dataset_loss = []
                dataset_acc = []
                for i in tqdm(range(length//3000 + 1)):
                    train_dataset = BatchedGraphDataset(args.train_path, i=i, transform=add_zeros)
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    train_loss = train(train_loader)
                    train_acc, _ = evaluate(train_loader, calculate_accuracy=True)
                    dataset_loss.append(train_loss)
                    dataset_acc.append(train_acc)
                    del train_dataset, train_loader
                    gc.collect()
                if epoch%10 == 0: 
                    with open(f'logs/log_training_datatset_{test_dir_name}.txt', 'w') as out:
                        out.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(dataset_loss):.4f}, Train Acc: {np.mean(dataset_acc):.4f}")
                torch.save(model.state_dict(), f'checkpoints/model_{test_dir_name}_epoch_{epoch + 1}.pth')
        # Evaluate and save test predictions
        
        test_dataset = BatchedGraphDataset(args.test_path, i=0, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        predictions = evaluate(test_loader, calculate_accuracy=False)
        test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

        # Save predictions to CSV
        
        output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
        output_df = pd.DataFrame({
            "id": test_graph_ids,
            "pred": predictions
        })
        output_df.to_csv(output_csv_path, index=False)
        print(f"Test predictions saved to {output_csv_path}")

    else:
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train dataset and loader (if train_path is provided)
        if args.train_path:
            train_dataset = GraphDataset(args.train_path, transform=add_zeros)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Training loop
            num_epochs = 2
            for epoch in range(num_epochs):
                train_loss = train(train_loader)
                train_acc, _ = evaluate(train_loader, calculate_accuracy=True)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    

        # Evaluate and save test predictions
        predictions = evaluate(test_loader, calculate_accuracy=False)
        test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

        # Save predictions to CSV
        test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
        output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
        output_df = pd.DataFrame({
            "id": test_graph_ids,
            "pred": predictions
        })
        output_df.to_csv(output_csv_path, index=False)
        print(f"Test predictions saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--jsonl", type=bool, default=False, help="Whether to use the jsonl format dataset")
    args = parser.parse_args()
    main(args)
