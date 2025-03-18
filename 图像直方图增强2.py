# tkinter 是一个缩写，它来自 “Tk interface”。
# 这里的 “Tk” 是指 “Tool Command Language” (Tcl) 的一个图形用户界面库，
# 而 “interface” 指的是这个库提供了一种接口或者方式，使得开发者可以用 Python 语言来创建图形用户界面。
import tkinter as tk    # 用于创建GUI

# filedialog 来自于 “file dialog” 这个英文短语。
# 在计算机编程和用户界面设计中，“file dialog” 是一个用于与用户交互以选择文件或目录的对话框。
# filedialog 模块是 Tkinter 库的一部分，
# 它提供了打开文件（Open File）、保存文件（Save File）和选择目录（Select Directory）等对话框的功能。
# 这些对话框通常包含了浏览文件系统、选择文件以及指定文件打开或保存选项的界面元素。
from tkinter import filedialog

import numpy as np

import matplotlib
# 指定matplotlib使用TkAgg作为后端，这样matplotlib的图形就可以嵌入到Tkinter窗口中。
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

# Python Imaging Library，用于处理图像。
from PIL import Image, ImageTk

# histogram:直方图
def histogram_equalization(image):
    """
    直方图均衡化函数
    :param image: 要处理的图像
    :return: 处理后的图像
    """
    # 计算图像的直方图
    # 直方图横坐标：灰度级，纵坐标：灰度级出现像素数量。
    # image.flatten(): 这是PIL或OpenCV图像对象的一个方法，它将图像数据转换为一个一维数组。这是因为np.histogram函数需要一维数据作为输入。
    # flatten的意思是把什么弄平。
    # np.histogram函数的第二个参数是箱数bins，这里表示直方图有256个箱，每个箱对应一个灰度值
    # 第三个参数用于指定包含直方图的数据范围，这里填[0,256]就把实际的0到255给包含完了。
    # 返回值hists是每个箱的计数或密度（取决于 density 参数，设置为True则为密度）。
    # 返回值bins是一个数组，包含箱的边缘值，长度比 hist 多一个，因为 bin 包含了每个箱的左闭右开区间的两个边界。
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积分布函数
    # cumsum 是 “cumulative sum” 的缩写，中文意思是“累加和”。
    cdf = hist.cumsum()

    # 归一化CDF
    cdf_normalized = cdf * hist.max() / cdf.max()


    # 使用累积分布函数的线性插值，计算新的像素值
    # 这个函数调用创建了一个被屏蔽的数组（masked array），它将数组 cdf 中所有等于0的元素屏蔽掉。
    cdf_m = np.ma.masked_equal(cdf, 0)  # 忽略掉0让我不是很理解
    # cdf_m =cdf

    # 通过这种方法把cdf_m原来的范围映射到了[0,255]
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # 调用将屏蔽数组 cdf_m 中的屏蔽元素（即原本等于0的元素）替换为0，并将数组的数据类型转换为无符号8位整数（uint8）
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # 公式是y = L*F(x)，L是自己定的，这里定为使之映射到了一个范围，之所以[0,255]：F(X)的值是灰度级数量，y的值是灰度级（像素的灰度级）
    # cdf其实就是L*F(x)，从归一化开始是乘以L的过程，之前就只是F(x)
    # 对于替换的索引：cdf在这里是一个256大小的数组，逐个扫描image，假设扫到image[i,j]，对应灰度级为m，就把m替换为cdf[m]
    # 总结来说，把一个灰度级范围单调线性映射到另一个灰度级范围。这里均衡化就使得某些灰度级没有那么极端。
    # 使用累积分布函数的变换值替换原始像素值
    image_equalized = cdf[image]    # 这个语法不是很理解
    return image_equalized

def show_images(original, equalized):
    # 显示原始图像和均衡化后的图像还有它们的直方图
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')   # cmap是颜色映射
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Equalized Image')
    plt.subplot(2, 2, 3)

    # ravel()是NumPy库中的一个函数，用于将多维数组展平成一维数组。
    # color用于设置直方图每个箱子的填充颜色。在这里，'k'代表黑色，所以直方图的每个箱子都将被填充为黑色。
    # edgecolor用于设置直方图每个箱子的边缘颜色。同样地，'k'代表黑色，所以直方图的每个箱子的边缘将被绘制为黑色。
    # ‘b’: 蓝色（blue）
    # ‘g’: 绿色（green）
    # ‘r’: 红色（red）
    # ‘c’: 青色（cyan）
    # ‘m’: 品红色（magenta）
    # ‘y’: 黄色（yellow）
    # ‘k’: 黑色（black）
    # ‘w’: 白色（white）
    # histtype='step'可以让看着稍微更连续一点？
    plt.hist(original.ravel(), bins=256, range=(0, 255), color='k', edgecolor='k')
    plt.title('Original Histogram')
    plt.subplot(2, 2, 4)
    plt.hist(equalized.ravel(), bins=256, range=(0, 255), color='red', edgecolor='red')
    plt.title('Equalized Histogram')
    plt.show()

def open_image():
    # 打开文件选择对话框
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    # 打开并显示图像
    image = Image.open(filepath).convert('L')  # 转换为灰度图像
    image_array = np.array(image)
    # 应用直方图均衡化
    image_equalized_array = histogram_equalization(image_array)
    # 显示图像
    show_images(image_array, image_equalized_array)

# 创建主窗口
root = tk.Tk()
root.title("Histogram Equalization")

# 创建一个按钮来打开文件选择对话框
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()  # 用于将组件添加到父窗口中

# 运行主循环
root.mainloop()
