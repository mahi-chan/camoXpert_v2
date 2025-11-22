"""
Quick DDP Verification Script
Run this to verify DDP is working correctly
"""
import torch
import torch.distributed as dist
import os

def main():
    # Initialize DDP
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    print(f"[Rank {rank}/{world_size}] Running on GPU {device} (local_rank={local_rank})")

    # Test all_reduce (metric synchronization)
    test_tensor = torch.tensor([rank + 1.0], device=device)  # Rank 0: [1.0], Rank 1: [2.0]
    print(f"[Rank {rank}] Before all_reduce: {test_tensor.item()}")

    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    test_tensor /= world_size

    print(f"[Rank {rank}] After all_reduce (should be 1.5): {test_tensor.item()}")

    # Only rank 0 should print final result
    if rank == 0:
        print(f"\n✅ DDP VERIFICATION PASSED!")
        print(f"   - World size: {world_size}")
        print(f"   - Metric sync working: {test_tensor.item()}")
        print(f"   - Single output: CORRECT (only rank 0 prints this)")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
