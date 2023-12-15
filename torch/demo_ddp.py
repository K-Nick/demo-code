import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os


# Simple model for demonstration
class DemoModel(nn.Module):

    def __init__(self):
        super(DemoModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def demo_ddp():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Setup for DistributedDataParallel
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)

        # Create a simple model and move it to GPU
        model = DemoModel().cuda()
        ddp_model = DDP(model, device_ids=[0])

        # Create dummy data and move it to GPU
        input = torch.randn(20, 10).cuda()
        target = torch.randint(0, 2, (20,)).cuda()

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

        # Forward pass
        outputs = ddp_model(input)
        loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("DDP Demo Completed Successfully")
    else:
        print("CUDA is not available. DDP requires GPUs.")


demo_ddp()
