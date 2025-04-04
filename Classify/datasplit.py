import os
import random

random.seed(42)

def data_split_classify(data_dir, output_dir, train_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    file_lists = {'train': [], 'val': [], 'test': []}

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)

        files = [class_dir + "/" + f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        random.shuffle(files)
        total = len(files)
        
        train_end = int(train_ratio * total)
        val_end = train_end + int((1-train_ratio)/2 * total)

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
    data_split_classify("./dataset/animal/", "./Datatxt/")
