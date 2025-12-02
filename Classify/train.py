import torch
from torch.utils.data import DataLoader
from dataloader import data_classify
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os 
import models
import matplotlib.pyplot as plt
import numpy as np


class Config:
    model_type = 'LeNet5'
    num_classes = 10

    data_dir = './Datatxt/'                      
    train_batch_size = 64
    val_batch_size = 8
    num_workers = 0
    use_cache = True

    optimizer_type = 'sgd'   # adamw, adam, sgd
    lr = 1e-2          # sgd->1e-2, adamw->1e-4, adam->1e-4
    epochs = 500  
    lr_min = 1e-8  
    lr_decay_epochs = 300  
    early_stop_patience = 20  
    early_stop_delta = 0.0001

    device = 'cpu'
    use_amp = True 

    loss_type = 'CrossEntropy'
    accumulation_steps = 1

    save_dir = './exp/' + model_type + '/checkpoints/'       
    use_checkpoints = False
    
    


def train():
    config = Config()
    if config.model_type == 'LeNet5':
        model = models.LeNet5(num_classes=config.num_classes)
    else:
        raise ValueError(f"cant supoort model: {config.model_type}")


    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if device.type == 'cpu':
        print("Warning: AMP is disabled because training is running on CPU.")
        use_amp = False 
    else:
        use_amp = config.use_amp

    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None 


    model.to(device)
    print(f"Using device: {device}")


    if config.loss_type == 'MSE':
        criterion = torch.nn.MSELoss()
    elif config.loss_type == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif config.loss_type == 'L1':
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {config.loss_type}")
    print(f"Using loss type: {config.loss_type}")
    
    if config.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
    print(f"Using optimizer type: {config.optimizer_type}")


    os.makedirs(Config.save_dir, exist_ok=True)

    if config.use_checkpoints:
        checkpoint = torch.load(config.save_dir+'last_model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        train_path = os.path.join(config.save_dir, 'train_loss.txt')
        val_path = os.path.join(config.save_dir, 'val_loss.txt')
        
        train_losses = []
        with open(train_path, 'r') as f:
            for line in f:
                train_losses.append(float(line.strip()))
                
        val_losses = []
        with open(val_path, 'r') as f:
            for line in f:
                val_losses.append(float(line.strip()))
        best_val_loss = min(val_losses)
        print(f"best val loss: {best_val_loss}")

    else:
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

    train_dataset = data_classify(path=config.data_dir, mode='train', transform=None,use_cache=config.use_cache)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset = data_classify(path=config.data_dir, mode='val', transform=None,use_cache=config.use_cache)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers)

    for epoch in range(Config.epochs):
        model.train()

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]',ncols=120)

        running_loss = 0.0
        step = 0
        
        for batch in train_progress:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps

            train_progress.set_postfix({'Loss': running_loss / (step + 1)})
            step += 1

        train_losses.append(running_loss / (step + 1))
        print(f'Epoch [{epoch+1}/{config.epochs}] train loss: {running_loss / (step + 1):.4f}')

        model.eval()
        val_running_loss = 0.0
        val_step = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_step += 1

        val_losses.append(val_running_loss / val_step)
        print(f'Epoch [{epoch+1}/{config.epochs}] val loss: {val_running_loss / val_step:.4f}')

        if (val_losses[-1] < best_val_loss) and (best_val_loss - val_losses[-1]) > config.early_stop_delta:
            best_val_loss = val_losses[-1]
            no_improve_epochs = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(config.save_dir, 'best_model.pth'))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                
            }, os.path.join(config.save_dir, 'last_model.pth'))
        else:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(config.save_dir, 'last_model.pth'))


        print(f'Epoch [{epoch+1}/{config.epochs}] Best Loss: {best_val_loss:.4f}')

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(Config.save_dir + 'loss_curve.png')
        plt.close()

        with open(os.path.join(config.save_dir, 'train_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write(f"{loss:.6f}\n")
        
        with open(os.path.join(config.save_dir, 'val_loss.txt'), 'w') as f:
            for loss in val_losses:
                f.write(f"{loss:.6f}\n")

        print(f"loss save: {config.save_dir+'loss_curve.png'}")
        print(f"train loss save: {config.save_dir+'train_loss.txt'}")
        print(f"val loss save: {config.save_dir+'val_loss.txt'}")

        data_iter = iter(val_loader)
 
        inputs, labels = next(data_iter)

        num_images = min(10, inputs.size(0))

        inputs, labels = inputs[:num_images].to(device), labels[:num_images].to(device)

        with torch.no_grad():
            with autocast(enabled=config.use_amp, dtype=torch.bfloat16): 
                outputs = model(inputs)
                

        _, predicted = torch.max(outputs.data, 1)

        rows = 2
        cols = (num_images + rows - 1) // rows 

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3)) 
        axes = axes.flatten()
        
        for i in range(len(inputs)):

            img = inputs[i].cpu().numpy().transpose((1, 2, 0)) 
            
            img = np.clip(img, 0, 1) 

            axes[i].imshow(img)
            
            title = f'True: {labels[i].item()}\nPred: {predicted[i].item()}'
            color = 'green' if predicted[i] == labels[i] else 'red'
            
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        
        #image_save_path = os.path.join(config.save_dir, f'predictions_epoch_{epoch+1}.png')
        image_save_path = os.path.join(config.save_dir, f'predictions.png')
        plt.savefig(image_save_path, dpi=200)
        plt.close(fig)
        
        print(f"Predictions saved to: {image_save_path}")
        model.train()



if __name__ == '__main__':
    train()