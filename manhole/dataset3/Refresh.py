import os
import xml.etree.ElementTree as ET
from PIL import Image

def update_xml_files(xml_folder, img_folder):
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        #img_subfolder = filename.split('_')[0]
        #img_folder = os.path.join(img_parent_folder, img_subfolder)
        img_path = os.path.join(img_folder, filename)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            
            root.find('size/width').text = str(width)
            root.find('size/height').text = str(height)
            
            tree.write(xml_path)
            print(f"Updated {xml_file} successfully.")
        else:
            print(f"Image file not found for {xml_file}.")

# 使用示例
xml_folder = './val/val_xml'
img_folder = './/val//img//'
update_xml_files(xml_folder, img_folder)
