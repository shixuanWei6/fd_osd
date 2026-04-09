# fileName: simulation/federated.py
import torch

class MultiClientFederatedServer:
    def __init__(self, template_model):
        """Initializes the server with the shape of the global adapter."""
        self.global_weights = {}
        for name, param in template_model.named_parameters():
            if "global_adapter" in name:
                self.global_weights[name] = param.data.clone().detach()

    def federated_averaging(self, clients):
        """Aggregates global_adapter weights from all clients using FedAvg."""
        print("\n[Federated Server] Aggregating Global LoRA weights from all clients...")
        
        with torch.no_grad():
            for name in self.global_weights.keys():
                # Extract the global adapter tensor from each client
                client_tensors = [client.model.get_parameter(name).data for client in clients]
                
                # Stack and compute the mean across all clients
                stacked_weights = torch.stack(client_tensors)
                avg_weight = torch.mean(stacked_weights, dim=0)
                
                # Update the central server state
                self.global_weights[name] = avg_weight.clone()

    def broadcast(self, clients):
        """Distributes the aggregated global weights back to all clients' global adapters."""
        print("[Federated Server] Broadcasting updated Global LoRA to clients...")
        with torch.no_grad():
            for client in clients:
                for name, param in client.model.named_parameters():
                    if "global_adapter" in name:
                        param.data.copy_(self.global_weights[name])