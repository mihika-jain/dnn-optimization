from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class SVDLinear(nn.Module):
    def __init__(self, original_linear, rank):
        super(SVDLinear, self).__init__()
        W = original_linear.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.rank = rank

        self.U = nn.Parameter(U[:, :rank], requires_grad=False)
        self.S = nn.Parameter(S[:rank], requires_grad=False)
        self.V = nn.Parameter(Vh[:rank, :], requires_grad=False)
        self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

    def forward(self, x):
        x = F.linear(x, self.V)
        
        x = x * self.S.unsqueeze(0)
        
        x = F.linear(x, self.U)
        
        x = x + self.bias
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy

def compress_model(model, rank):
    model.fc1 = SVDLinear(model.fc1, rank)
    model.fc2 = SVDLinear(model.fc2, rank)
    # Skip fc3 (small output layer) for compression
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with SVD Compression')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--svd-rank', type=int, default=None,
                        help='Rank for SVD compression. If not set, no compression applied.')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Save baseline model parameters
    original_params = count_parameters(model)

    # Apply SVD compression if specified
    if args.svd_rank is not None:
        print(f"Compressing model with SVD rank={args.svd_rank}...")
        model = compress_model(model, args.svd_rank)
        compressed_params = count_parameters(model)
        compression_ratio = original_params / compressed_params
        print(f"Compression Ratio: {compression_ratio:.2f}")

        # Measure runtime after compression
        start_time = time.time()
        accuracy = test(model, device, test_loader)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Post-compression Accuracy: {accuracy:.2f}%")
        print(f"Measured Runtime: {runtime:.4f} seconds")

        # Summarize
        print("\n=== Results ===")
        print(f"SVD Rank: {args.svd_rank}")
        print(f"Compression Ratio: {compression_ratio:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Runtime: {runtime:.4f} seconds")
        print("================\n")
    else:
        print("No SVD compression applied.")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_fc.pt")

if __name__ == '__main__':
    main()
