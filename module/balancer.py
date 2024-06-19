import torch
import torch.nn as nn

class AdaptiveLossWeighting(nn.Module):
    def __init__(self, num_losses, temperature=1, k=10):
        super(AdaptiveLossWeighting, self).__init__()
        self.num_losses = num_losses
        self.prev_losses = None  
        self.weights = torch.ones(num_losses)  
        self.t = temperature
        self.k = k

    def forward(self, current_losses):
        max_loss = current_losses.max()
        magnitude_adjustment = max_loss / (current_losses + 1e-8)
        adjusted_losses = current_losses * magnitude_adjustment

        if self.prev_losses is None:
            self.prev_losses = adjusted_losses.detach_()
            return self.weights


        loss_rates = adjusted_losses / (self.prev_losses + 1e-8)

        numerator = self.k * torch.exp(loss_rates / self.t)

        denominator = torch.sum(torch.exp(loss_rates / self.t))

        self.weights = (numerator / denominator) * magnitude_adjustment
        
        self.prev_losses = adjusted_losses.detach_()

        return self.weights


if __name__ == '__main__':
    # demo
    adaptive_loss_weighting = AdaptiveLossWeighting(num_losses=2)

    for step in range(200):
        current_losses = torch.tensor([1.0 / (step + 10000), 200000.0 / (step + 1)])
        weights = adaptive_loss_weighting.update_loss_weights(current_losses)
        print(f"Step {step}, Weights: {weights}, Loss: {torch.sum(current_losses*weights)}")
