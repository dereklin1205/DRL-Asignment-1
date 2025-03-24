import torch
import torch.nn as nn
import os

# Simple script to create a model file needed by the student_agent.py

# Create model directory
os.makedirs('trained_models', exist_ok=True)

# Define a simple model with the same structure used in student_agent.py
class SimpleModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=6):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(8, output_dim)
        )
        self.action_head = nn.Softmax(dim=-1)
    
    def forward(self, x):
        logits = self.network(x)
        return self.action_head(logits)

# Create model
model = SimpleModel()

# Save model to the expected path
torch.save(model.state_dict(), 'trained_models/policy_model_04.pth')

print("Created model file at trained_models/policy_model_04.pth")