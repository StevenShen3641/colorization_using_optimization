import cv2
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

NUM_IN_WINDOW = 9
"""
NUM_IN_WINDOW: 一个窗格内的像素个数
"""


def colorization(p_original, p_marked, p_type):
    """
    优化上色算法
    :param p_original: 原始黑白图片，格式为uint8
    :param p_marked: 带标记的图片，格式为uint8
    :param p_type: 比重函数的选择
    :return: 上色后的图片
    """

    # 先转换为32位浮点类型方便计算
    p_original = p_original.astype(np.float32) / 255
    p_marked = p_marked.astype(np.float32) / 255

    # 转换为YUV格式
    p_original = cv2.cvtColor(p_original, cv2.COLOR_BGR2YUV)
    p_marked = cv2.cvtColor(p_marked, cv2.COLOR_BGR2YUV)

    # 判断像素值是否被标记
    # abs()防止原数组三个数一正一负相加刚好等于0的情况
    is_marked = abs(p_original - p_marked).sum(2) != 0

    # 创建相同大小的零矩阵，存放原始黑白图片的Y值和标记后图片的UV值
    YUV_comb = np.zeros(p_original.shape)
    YUV_comb[:, :, 0] = p_original[:, :, 0]
    YUV_comb[:, :, 1] = p_marked[:, :, 1]
    YUV_comb[:, :, 2] = p_marked[:, :, 2]

    # 获取宽高和图像大小
    height = p_original.shape[0]
    width = p_original.shape[1]
    image_size = height * width

    # 建立下标矩阵
    # order='F'，为竖着读取竖着填充
    indices = np.arange(image_size).reshape(height, width, order='F').copy()

    # 最大像素接触数量为一个窗格的数量乘上图像像素个数
    max_pxls = image_size * NUM_IN_WINDOW

    # 按最大像素接触数量建立max_pxls * max_pxls稀疏矩阵
    # row_indices存放权重系数及其中心像素对应的线性方程组的行下标
    # col_indices存放权重系数及其中心像素在该线性方程组中的列下标
    # values存放权重系数及其中心像素在线性方程中的系数1
    # 三者一一对应
    row_indices = np.zeros(max_pxls)
    col_indices = np.zeros(max_pxls)
    values = np.zeros(max_pxls)

    index = 0  # 存放权重系数相关值的三个数组的下标
    current_pxl_index = 0  # 当前中心像素下标

    # 遍历图中每个像素，计算对应各个方向权重
    for col in range(width):
        for row in range(height):

            # 如果没有被标记
            if not is_marked[row, col]:
                window_index = 0  # 当前3 * 3窗格内的下标
                window_intst = np.zeros(NUM_IN_WINDOW)  # 存放每个下标对应的Y值

                # 遍历窗格内的各个像素，记录对应Y值
                for lcl_col in range(max(0, col - 1), min(col + 2, width)):
                    for lcl_row in range(max(0, row - 1), min(row + 2, height)):

                        # 不为中心像素时
                        if lcl_col != col or lcl_row != row:
                            row_indices[index] = current_pxl_index  # 记录周围像素对应中心像素的下标
                            col_indices[index] = indices[lcl_row, lcl_col]  # 记录周围像素下标
                            window_intst[window_index] = YUV_comb[lcl_row, lcl_col, 0]  # 记录周围像素Y值

                            index += 1
                            window_index += 1

                center_intst = YUV_comb[row, col, 0].copy()  # 中心像素Y值
                window_intst[window_index] = center_intst  # 记录中心像素Y值

                # 计算方差（包含中心）
                mean = np.mean(window_intst[0:window_index + 1])
                variance = np.mean((window_intst[0:window_index + 1] - mean) ** 2)
                window_weight = np.zeros(NUM_IN_WINDOW - 1)  # 存放对应周围像素的比例系数

                # 采用正态分布函数
                if p_type == 'Y':
                    # 非零时，降低方差，给予相同或相似的像素更多比重
                    # 为零时，比重都相同，不受方差影响，因此方差可以为任何值，这里赋值为1
                    sigma_sqr = 1 if variance == 0 else 0.3 * variance

                    # 计算比例系数
                    window_weight[0:window_index] = np.exp(
                        -((window_intst[0:window_index] - center_intst) ** 2) / (2 * sigma_sqr))

                # 采用相关性函数
                elif p_type == 'N':
                    # 非零时，
                    # 法一：由于相关性函数对数据十分敏感，所以增大方差，降低斜率
                    # sigma_sqr = 1 if variance == 0 else 1.5 * variance
                    # 法二：在保证数据有效
                    # （不由于舍入规则使一些数据变为0）的情况下，减小负数部分的比重
                    # 防止出现YUV转RGB时溢出导致出现大面积单色块的情况
                    sigma_sqr = 1 if variance == 0 else 1.00001 * variance

                    # 计算比例系数
                    window_weight[0:window_index] = 1 + (window_intst[0:window_index] - mean) * (
                            center_intst - mean) / sigma_sqr

                    # 减小负数部分的比重
                    for i in range(window_index):
                        if window_weight[i] < 0:
                            window_weight[i] = window_weight[i] / 10

                # 输入错误
                else:
                    return

                # 使比例系数平衡到1
                window_weight[0:window_index] = window_weight[0:window_index] / np.sum(
                    window_weight[0:window_index])

                values[index - window_index:index] = -window_weight[0:window_index]  # 记录比例系数

            # 记录当前的中心像素
            row_indices[index] = current_pxl_index
            col_indices[index] = current_pxl_index
            values[index] = 1
            index += 1
            current_pxl_index += 1

    # 去除空元素位
    values = values[0:index]
    col_indices = col_indices[0:index]
    row_indices = row_indices[0:index]

    # 建立稀疏矩阵，大小取image_size * image_size，建立等式右侧值
    A = csr_matrix((values, (row_indices, col_indices)), shape=(image_size, image_size))
    b = np.zeros(image_size)

    # 将判断标记像素数组变成一维
    is_marked = is_marked.reshape(image_size, order='F')

    # 返回判断标记像素数组中非零元素的下标索引
    marked_indices = np.nonzero(is_marked)

    # 新建要输出的将要上色的图片
    colorized = np.zeros(YUV_comb.shape)
    # 先复制亮度值
    colorized[:, :, 0] = YUV_comb[:, :, 0]

    # 分别计算U和V
    for i in [1, 2]:
        chrominance = YUV_comb[:, :, i].reshape(image_size, order='F').copy()
        b[marked_indices] = chrominance[marked_indices]  # 给标记的像素赋值
        res = spsolve(A, b)
        colorized[:, :, i] = res.reshape(height, width, order='F').copy()

    # 转换BGR，转换前要满足float32的类型
    colorized = colorized.astype(np.float32)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_YUV2BGR)

    return colorized


def main():
    # 导入图像
    original_file = input('Enter original file name: ')
    marked_file = input('Enter marked file name: ')
    func_type = input("Enter 'Y' for exponential weighting function. "
                      "Enter 'N' for correlation weighting function: ")
    original = cv2.imread(original_file)
    marked = cv2.imread(marked_file)

    # 处理
    start_time = time.time()
    colorized_pic = colorization(original, marked, func_type)

    # 输入无效
    if colorized_pic is None:
        print('Invalid input for weighting function!')
        return
    end_time = time.time()
    print('It spends {time} seconds.'.format(time=end_time - start_time))

    # 展示
    cv2.imshow('tgt', colorized_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
