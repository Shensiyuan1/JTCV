import torch
import torchvision
import os
import random
from torch.utils.data import Dataset,DataLoader
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

class data_classify(Dataset):
    def __init__(self,path,transform = None,mode = 'train',use_cache=False):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.pth = path
        self.input = []
        self.labels = []
        self.use_cache = use_cache
        self.cache = []
        
        if mode == 'train':    #read train data dir from train.txt
            with open(self.pth+'train.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.input.append(os.path.join(line))
            with open(self.pth+'train_label.txt', 'r', encoding='utf-8') as file: #read train label from train_label.txt
                for line in file:
                    line = line.replace('\n','')
                    self.labels.append(int(line))
        
        if mode =='val':
            with open(self.pth+'val.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.input.append(os.path.join(line))
            with open(self.pth+'val_label.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.labels.append(int(line))

        if mode =='test':
            with open(self.pth+'test.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.input.append(os.path.join(line))
            with open(self.pth+'test_label.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.labels.append(int(line))

        if self.use_cache and len(self.input) > 0:
            cache_path = os.path.join(self.pth, f"{mode}_cache.pt")

            if os.path.exists(cache_path):
                print(f"[{mode}]: Loading persistent cache from {cache_path}...")
                cached_data = torch.load(cache_path)
                self.cache = cached_data['images']
                self.labels = cached_data['labels'] 
                self.input = [] 
                print(f"[{mode}]: Persistent cache loaded. Size: {len(self.cache)}")
            else: 
                print(f"[{mode}]: Caching {len(self.input)} images into memory...")
                for path in tqdm(self.input, desc=f"Caching {mode} data", ncols=80, leave=True):
                    try:
                        img = Image.open(path).convert('RGB')
                        img.load()
                        img_array = np.array(img)
                        self.cache.append(img_array)
                        img.close()

                    except Exception as e:
                        print(f"Error loading image {path} during caching: {e}")
                        self.cache.append(None) 
                
                self.input = [p for i, p in enumerate(self.input) if self.cache[i] is not None]
                self.labels = [l for i, l in enumerate(self.labels) if self.cache[i] is not None]
                self.cache = [c for c in self.cache if c is not None]
                
                print(f"[{mode}]: Caching complete. Actual size: {len(self.cache)}")

                if len(self.cache) > 0:
                    cache_to_save = {'images': self.cache, 'labels': self.labels}
                    torch.save(cache_to_save, cache_path)
                    print(f"[{mode}]: Persistent cache saved successfully!")
 
                print(f"[{mode}]: Caching complete. Actual size: {len(self.cache)}")
                  

    def __len__(self):
        if self.use_cache:
            return len(self.cache)
        else:
            return len(self.input)

    def __getitem__(self,idx):
        # img = cv2.imread(self.input[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img.astype('uint8'), 'RGB') 

        if self.use_cache:
            img = self.cache[idx].copy()
            img = Image.fromarray(img)
        else:
            img = Image.open(self.input[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        else:
            img = torchvision.transforms.functional.to_tensor(img)
        label = torch.tensor(self.labels[idx])

        return img,label

def data_split_classify(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1):

    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)    #create output_dir if not exist
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]) #get class name

    file_lists = {'train': [], 'val': [], 'test': []}

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)

        files = [class_dir + "/" + f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        random.shuffle(files)
        total = len(files)
        
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        file_lists['train'].extend([(f, class_idx) for f in files[:train_end]])
        file_lists['val'].extend([(f, class_idx) for f in files[train_end:val_end]])
        file_lists['test'].extend([(f, class_idx) for f in files[val_end:]])

    for split in ['train', 'val', 'test']:
        with open(os.path.join(output_dir, f"{split}.txt"), 'w') as f_path, \
            open(os.path.join(output_dir, f"{split}_label.txt"), 'w') as f_label:
            for path, label in file_lists[split]:
                f_path.write(f"{path}\n") 
                f_label.write(f"{label}\n")  

    print("finish")



if __name__ == "__main__":

    data_split_classify("./dataset/mnist/", "./Datatxt/")

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor()])
    dataset = data_classify(path="./Datatxt/",mode="train",transform=None)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    for img,label in dataloader:
        print(img.shape,label.shape)
        break
    
   
