#!/usr/bin/env python3
"""
Test script for DDP (Distributed Data Parallel) training.

Usage:
    # Single GPU (no DDP)
    python test_ddp.py

    # Multi-GPU with DDP (using torchrun)
    torchrun --nproc_per_node=4 test_ddp.py

    # Multi-GPU with DDP (manual environment variables)
    WORLD_SIZE=4 RANK=0 LOCAL_RANK=0 python test_ddp.py  # Process 0
    WORLD_SIZE=4 RANK=1 LOCAL_RANK=1 python test_ddp.py  # Process 1
    WORLD_SIZE=4 RANK=2 LOCAL_RANK=2 python test_ddp.py  # Process 2
    WORLD_SIZE=4 RANK=3 LOCAL_RANK=3 python test_ddp.py  # Process 3
"""

import os
import sys

# Add parent directory to path to import latentwire
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import our DDP manager from train.py
from latentwire.train import DDPManager, ElasticGPUConfig, initialize_ddp_from_elastic_config


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random data for testing
        return torch.randn(10), torch.randint(0, 10, (1,)).item()


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_ddp():
    """Test DDP functionality."""

    print("="*60)
    print("DDP Test Script")
    print("="*60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, testing CPU mode only", flush=True)
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    # Initialize DDP manager
    ddp_manager = DDPManager()

    # Try to initialize DDP
    if ddp_manager.initialize():
        print(f"\n✅ DDP initialized successfully!")
        print(f"  World size: {ddp_manager.world_size}")
        print(f"  Rank: {ddp_manager.rank}")
        print(f"  Local rank: {ddp_manager.local_rank}")
        print(f"  Device: {ddp_manager.device}")
        print(f"  Is main process: {ddp_manager.is_main_process}")
    else:
        print("\n⚠️ DDP not initialized (single process mode)")
        print("  To test DDP, run with torchrun:")
        print("    torchrun --nproc_per_node=2 test_ddp.py")

    # Create model
    model = SimpleModel()
    ddp_manager.print(f"\nModel created: {model.__class__.__name__}")

    # Move model to device and wrap with DDP if initialized
    model = model.to(ddp_manager.device if ddp_manager.initialized else device)
    if ddp_manager.initialized:
        model = ddp_manager.wrap_model(model)
        ddp_manager.print("Model wrapped with DDP")

    # Create dataset and dataloader
    dataset = DummyDataset(size=100)

    if ddp_manager.initialized:
        # Use DDP-aware dataloader
        dataloader = ddp_manager.get_dataloader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=0
        )
        ddp_manager.print("Created DDP-aware DataLoader with DistributedSampler")
    else:
        # Regular dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=0
        )
        print("Created regular DataLoader")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop (just 1 epoch for testing)
    ddp_manager.print("\nStarting training loop...", flush=True)

    model.train()
    for epoch in range(2):
        # Set epoch for distributed sampler
        ddp_manager.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to device
            data = data.to(ddp_manager.device if ddp_manager.initialized else device)
            target = target.to(ddp_manager.device if ddp_manager.initialized else device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Print progress (only from main process)
            if batch_idx % 5 == 0:
                ddp_manager.print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}", flush=True)

        # Calculate average loss
        avg_loss = total_loss / num_batches

        # Synchronize loss across all processes if using DDP
        if ddp_manager.initialized:
            loss_tensor = torch.tensor(avg_loss).to(ddp_manager.device)
            ddp_manager.all_reduce(loss_tensor)
            loss_tensor /= ddp_manager.world_size
            avg_loss = loss_tensor.item()

        ddp_manager.print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}", flush=True)

        # Synchronize all processes
        ddp_manager.barrier()

    # Test ElasticGPUConfig integration
    ddp_manager.print("\n" + "="*60)
    ddp_manager.print("Testing ElasticGPUConfig integration")
    ddp_manager.print("="*60)

    if torch.cuda.is_available():
        elastic_config = ElasticGPUConfig(base_batch_size=64)
        if ddp_manager.is_main_process:
            elastic_config.print_config()

        # Try to initialize DDP from elastic config
        # (This would normally be done at the start of training)
        if not ddp_manager.initialized and elastic_config.config.get('ddp'):
            ddp_manager.print("ElasticGPUConfig suggests DDP but it's not initialized")
            ddp_manager.print("Run with torchrun to enable DDP")

    # Cleanup
    ddp_manager.print("\n✅ Test completed successfully!", flush=True)
    ddp_manager.cleanup()


if __name__ == "__main__":
    test_ddp()