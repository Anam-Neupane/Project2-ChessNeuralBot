# Wishful thinking 

import torch 
import torch.nn as nn
import torch.nn.functional as F 

class ResBlock(nn.Module): 
    """
    A residual block: two conv layers with a skip connection. 
    The skip connection lets gradients flow directly, enabling deeper networks. 
    """
    
    def __init__(self, channels: int = 128): 
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x   # Save the input for the skip 
        
        x = F.relu(self.bn1(self.conv1(x)))  # First conv layer
        x = self.bn2(self.conv2(x))             # Second conv layer (No ReLU yet)
        
        return F.relu(x + residual)         # add skip, then ReLU
    
class ChessNet(nn.Module): 
    """
    The full chess neural network. 
    
    Input: (batch, 17, 8, 8) - our board tensor
    Output: policy (batch, 4096) - raw logits for each possible from -> to move 
            value (batch, 1)    - position evaluation in [-1, +1]
                                    +1 = White is winning, -1 = Black is winning
    
    Architecutre overview: 
        Stem (17 -> 128 channels) -> 6 residual blocks
        -> policy head: 128=>2 conv -> flatten -> 128 -> 4096 linear
        -> value head: 128=>1 conv -> flatten -> 64 -> 1 linear with tanh
    """
    
    def __init__(self, num_res_blocks: int = 6, channels: int = 128): 
        super().__init__()
        
        # Stem: first convolution to expand from 17 channels to 'Channels'
        self.stem = nn.Sequential(
            nn.Conv2d(17, channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(channels), 
            nn.ReLU(), 
        )
        
        # Residual tower 
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
    
        # Policy head 
        # Compress spatial info into a flat move-probability vector 
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False) 
        self.policy_bn = nn.BatchNorm2d(2) 
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) # 128 inputs -> 4096 move logits
        
        # Value head 
        # Compress the board into a single scalar [-1, +1]
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False) 
        self.value_bn = nn.BatchNorm2d(1) 
        self.value_fc1 = nn.Linear(8 * 8, 64) 
        self.value_fc2 = nn.Linear(64, 1) 
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        # Shared feature extraction
        x = self.stem(x) 
        x = self.res_blocks(x) 
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) 
        
        policy = self.policy_fc(p) 
        
        # Value head 
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) 
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value
    
if __name__ == "__main__":
    model = ChessNet(num_res_blocks=6, channels=128)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}") 
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 17, 8, 8) 
    policy, value = model(dummy_input) 
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    print(f"Value range:        [{value.min():.3f}, {value.max():.3f}]")
    
    