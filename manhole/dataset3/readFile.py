import os

folder_path = './test2/txt' 
txt_file_path = './test.txt'

file_names = os.listdir(folder_path)

with open(txt_file_path, 'w') as f:
    for file_name in file_names:
        name = os.path.splitext(file_name)[0]
        f.write(name + '\n')
