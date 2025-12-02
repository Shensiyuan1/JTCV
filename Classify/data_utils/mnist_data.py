import pandas as pd
import numpy as np
from PIL import Image
import os

#mnist_csv -> "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
def csv_to_images_mnist(csv_file, output_dir,count=0):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    labels =  df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values


    for i, (label, pixel_row) in enumerate(zip(labels, pixels)):
        pixel_row = np.clip(pixel_row, 0, 255).astype(np.uint8)
        
        img_array = pixel_row.reshape(28, 28)
        
        img = Image.fromarray(img_array, mode='L')  
        

        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        img_path = os.path.join(label_dir, f"{count}.bmp")
        img.save(img_path)
        count += 1

        if (i + 1) % 1000 == 0:
            print(f"get {count} / {len(df)}  img")


if __name__ == '__main__':
    csv_to_images_mnist('../dataset/mnist_test.csv','../dataset/mnist/',count=60000)
