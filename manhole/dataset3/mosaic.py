import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

# 读取图片
def load_image(img_files,index):
    '''
    根据index读取图片
    '''
    path=img_files[index]
    img=cv2.imread(path)
    # 图片原始H,W
    h0,w0=img.shape[:2]
    # 较长的边缩放到640,另一边也等比例缩放
    r=640/max(h0,w0)
    if r!=1:
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    return img, (h0, w0), img.shape[:2]

# 显示图片
def show(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(img)
    plt.show()
    plt.pause(5)

# 标签信息
# class,x,y,,w,h
top_left=np.array([[1,0.502 ,0.5343082114735658, 0.625, 0.4109486314210724]],dtype=np.float32)
top_right=np.array([[1, 0.5408333333333334, 0.5316558441558442, 0.6816666666666668, 0.5275974025974026]],dtype=np.float32)
down_left=np.array([[1, 0.41393849206349204, 0.34160052910052907, 0.4350198412698412, 0.6818783068783069]],dtype=np.float32)
down_right=np.array([[1, 0.40125391849529785, 0.7024410774410774, 0.7628004179728318, 0.595117845117845]],dtype=np.float32)
total_labels=[top_left,top_right,down_left,down_right]

if __name__=="__main__":
    # 定义变量
    s=640
    mosaic_border=[-320,-320]
    # 创建画布H*W*C=1280*1280*3
    img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    # 画布中心区域产生中心点
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]

    # 四张图片索引
    indices=[0,1,2,3]
    # 四张图片的路径
    img_files=["6.jpg","7.jpg","8.jpg","9.jpg"]
    # Mosaic数据增强
    # 存放四张图像的标签
    labels4=[]
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(img_files, index)
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        # 将图片放置在画布相应位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        # 调整标签
        padw = x1a - x1b
        padh = y1a - y1b
        x=total_labels[index]
        labels=x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)
    if len(labels4):
        # 将四张图像的标签进行拼接
        labels4 = np.concatenate(labels4, 0)
        # 裁剪标签,保证数据取值在0~2*s
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
    # 标签尺寸为n*5,n表示bbox个数
    print(labels4.shape)
    # 将bbox绘制在画布上
    for i in range(len(labels4)):
        box=labels4[i]
        # 左上角,右下角
        c1,c2=(int(box[1]),int(box[2])),(int(box[3]),int(box[4]))
        cv2.rectangle(img4,c1,c2,(0,255,0),thickness=4,lineType=cv2.LINE_AA)
    # 显示画布
    show(img4) 
