
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DixonDNN(nn.Module):
    """
    Deep Neural Network following Dixon et al. 2017
    
    Architecture:
    - Input: n_features (555 in our case)
    - Hidden 1: 1000 neurons + Sigmoid
    - Hidden 2: 900 neurons + Sigmoid  
    - Hidden 3: 800 neurons + Sigmoid
    - Hidden 4: 700 neurons + Sigmoid
    - Output: 3 neurons + Softmax (for {-1, 0, +1})
    
    Loss: Cross-entropy (Equation 1 in paper)
    Optimizer: SGD with adaptive learning rate
    """
    
    def __init__(self, input_dim, n_classes=3, dropout_rate=0.2):
        super(DixonDNN, self).__init__()
        
        # Architecture from Dixon paper
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 900)
        self.fc3 = nn.Linear(900, 800)
        self.fc4 = nn.Linear(800, 700)
        self.fc5 = nn.Linear(700, n_classes)
        
        # Activation (Dixon uses sigmoid, we can also try ReLU)
        self.activation = nn.Sigmoid()
        
        # Dropout for regularization (not in original paper, but helps)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights: Gaussian N(0, 0.01) - Dixon method
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Gaussian N(0, 0.01) like Dixon"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)  # Dixon: bias = 1
    
    def forward(self, x):
        """Forward pass through network"""
        # Layer 1
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 4
        x = self.fc4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output layer (logits)
        x = self.fc5(x)
        
        return x  # Softmax applied in loss function


class DixonTrainer:
    """
    Training engine following Algorithm 2 from Dixon paper
    - Mini-batch SGD
    - Adaptive learning rate (halve if loss doesn't decrease)
    - Cross-entropy loss
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss (cross-entropy)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Loss
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        return val_loss / len(val_loader), accuracy
    
    def train(self, train_loader, val_loader, n_epochs=50, 
              initial_lr=0.1, lr_patience=5):
        """
        Full training loop with adaptive learning rate
        Following Dixon Algorithm 2
        """
        # Cross-entropy loss (Equation 1)
        criterion = nn.CrossEntropyLoss()
        
        # SGD optimizer (Dixon uses SGD, not Adam)
        optimizer = optim.SGD(self.model.parameters(), lr=initial_lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("=" * 60)
        print("TRAINING DIXON DNN")
        print("=" * 60)
        print(f"Initial learning rate: {initial_lr}")
        print(f"Epochs: {n_epochs}")
        print(f"Mini-batch size: {train_loader.batch_size}")
        print("=" * 60)
        
        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}% | "
                      f"LR: {optimizer.param_groups[0]['lr']:.4f}")
            
            # Adaptive learning rate (Dixon: halve if no improvement)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
                if patience_counter >= lr_patience:
                    # Halve learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"→ Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
                    patience_counter = 0
            
            # Early stopping if LR too small
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print(f"\nEarly stopping: Learning rate too small")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("=" * 60)
        print("✓ Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 60)
        
        return self.train_losses, self.val_losses
    
    def predict(self, test_loader):
        """Make predictions on test set"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Probabilities
                probs = torch.softmax(output, dim=1)
                
                # Predictions
                pred = output.argmax(dim=1)
                
                all_preds.append(pred.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        return (np.concatenate(all_preds), 
                np.concatenate(all_probs),
                np.concatenate(all_targets))
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Model loaded from {path}")
