import torch

#Temperature determines significance of unimodality
class ORCU(torch.nn.Module):
    def __init__(self, temperature):
        super(ORCU, self).__init__()
        self.t = torch.tensor(temperature)
        self.soft_targets = None

    def I(self, r):
        if r <= (-1/(self.t**2)):
            return (-1/self.t) * torch.log(-r)
        else:
            return (self.t * r) - ((1/self.t) * torch.log(1/(self.t**2))) + (1/self.t)
    
    #Model output should be in logits, and target a one-hot vector
    def forward(self, output, target):
        device = output.device
        self.t.to(device)
        n_classes = target.size()[1]
        b_size = target.size()[0]

        if self.soft_targets is None:
            dist_mat = torch.tensor([ list(range(n_classes)) for _ in range(n_classes) ], dtype=torch.float, device=device)
            dist_mat = dist_mat - dist_mat[0].reshape((n_classes, 1))
            dist_mat = torch.abs(dist_mat)
            dividends = torch.exp(-dist_mat)
            divisors = torch.sum(dividends, 1, keepdim=True)
            self.soft_targets = dividends / divisors

        class_labels = torch.argmax(target, 1)
        batch_soft_targets = self.soft_targets[class_labels]

        ce_loss = torch.nn.functional.cross_entropy(output, batch_soft_targets)

        reg = torch.tensor([0.0], device=device)
        for n in range(b_size):
            for k in range(n_classes-1):
                if k < class_labels[n]:
                    reg += self.I(output[n][k] - output[n][k+1])
                else:
                    reg += self.I(output[n][k+1] - output[n][k])
        reg /= b_size

        return ce_loss + reg