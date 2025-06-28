import torch
import torch.nn as nn
import torch.nn.functional as F

class CFDM(nn.Module):
    def __init__(self, in_channels, head_dim, num_heads):
        super(CFDM, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.query1 = nn.Linear(in_channels, num_heads * head_dim)
        self.key1 = nn.Linear(in_channels, num_heads * head_dim)
        self.value1 = nn.Linear(in_channels, num_heads * head_dim)
        self.query2 = nn.Linear(in_channels, num_heads * head_dim)
        self.key2 = nn.Linear(in_channels, num_heads * head_dim)
        self.value2 = nn.Linear(in_channels, num_heads * head_dim)
        self.output_linear = nn.Linear(num_heads * head_dim, in_channels)

    def forward(self, F1, F2):
        B, C, H, W = F1.shape
        F1_flat = F1.view(B, C, -1).permute(0, 2, 1)
        F2_flat = F2.view(B, C, -1).permute(0, 2, 1)
        Q1 = self.query1(F1_flat)
        K1 = self.key1(F1_flat)
        V1 = self.value1(F1_flat)
        Q2 = self.query2(F2_flat)
        K2 = self.key2(F2_flat)
        V2 = self.value2(F2_flat)
        Q1 = Q1.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K1 = K1.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V1 = V1.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Q2 = Q2.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K2 = K2.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V2 = V2.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention_weights_1 = torch.matmul(Q1, K2.transpose(-2, -1)) * self.scale
        attention_weights_1 = F.softmax(attention_weights_1, dim=-1)
        cross_attended_1 = torch.matmul(attention_weights_1, V2)
        attention_weights_2 = torch.matmul(Q2, K1.transpose(-2, -1)) * self.scale
        attention_weights_2 = F.softmax(attention_weights_2, dim=-1)
        cross_attended_2 = torch.matmul(attention_weights_2, V1)
        cross_attended_1 = cross_attended_1.permute(0, 2, 1, 3).contiguous().view(B, -1, self.num_heads * self.head_dim)
        cross_attended_2 = cross_attended_2.permute(0, 2, 1, 3).contiguous().view(B, -1, self.num_heads * self.head_dim)
        fused_F1 = self.output_linear(cross_attended_1).permute(0, 2, 1).view(B, C, H, W)
        fused_F2 = self.output_linear(cross_attended_2).permute(0, 2, 1).view(B, C, H, W)
        fused_feature =  fused_F1 + fused_F2
        return fused_feature
