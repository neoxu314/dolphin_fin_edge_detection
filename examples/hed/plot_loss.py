'''
Plot a line graph of different loss values produced by HED training

Loss values:
1. Overall loss
2. Dsn1 loss
3. Dsn2 loss
4. Dsn3 loss
5. Dsn4 loss
6. Dsn5 loss
7. Fuse loss
'''

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np


AVE_DEN = 20
STEP_SIZE = 20


def plot_ave_loss(path_to_loss_file):
    file = open(path_to_loss_file)
    file_content = file.readlines()
    file_content_list = [x.strip() for x in file_content]

    iteration = []
    loss_a = []
    loss_dsn1 = []
    loss_dsn2 = []
    loss_dsn3 = []
    loss_dsn4 = []
    loss_dsn5 = []
    loss_fuse = []

    for item in file_content_list:
        item = item.split(' ')
        iteration.append(int(item[0]))
        loss_a.append(float(item[1]))
        loss_dsn1.append(float(item[2]))
        loss_dsn2.append(float(item[3]))
        loss_dsn3.append(float(item[4]))
        loss_dsn4.append(float(item[5]))
        loss_dsn5.append(float(item[6]))
        loss_fuse.append(float(item[7]))

    loss_a_ave = []
    loss_dsn1_ave = []
    loss_dsn2_ave = []
    loss_dsn3_ave = []
    loss_dsn4_ave = []
    loss_dsn5_ave = []
    loss_fuse_ave = []

    loss_a_sum = 0
    loss_dsn1_sum = 0
    loss_dsn2_sum = 0
    loss_dsn3_sum = 0
    loss_dsn4_sum = 0
    loss_dsn5_sum = 0
    loss_fuse_sum = 0

    count = 1
    for loss_a_item, loss_dsn1_item, loss_dsn2_item, loss_dsn3_item, loss_dsn4_item, loss_dsn5_item, loss_fuse_item in zip(loss_a, loss_dsn1, loss_dsn2, loss_dsn3, loss_dsn4, loss_dsn5, loss_fuse):
        if count % AVE_DEN != 0:
            count += 1
            loss_a_sum += loss_a_item
            loss_dsn1_sum += loss_dsn1_item
            loss_dsn2_sum += loss_dsn2_item
            loss_dsn3_sum += loss_dsn3_item
            loss_dsn4_sum += loss_dsn4_item
            loss_dsn5_sum += loss_dsn5_item
            loss_fuse_sum += loss_fuse_item
        else:
            loss_a_sum += loss_a_item
            loss_dsn1_sum += loss_dsn1_item
            loss_dsn2_sum += loss_dsn2_item
            loss_dsn3_sum += loss_dsn3_item
            loss_dsn4_sum += loss_dsn4_item
            loss_dsn5_sum += loss_dsn5_item
            loss_fuse_sum += loss_fuse_item

            loss_a_ave.append(loss_a_sum / float(AVE_DEN))
            loss_dsn1_ave.append(loss_dsn1_sum / float(AVE_DEN))
            loss_dsn2_ave.append(loss_dsn2_sum / float(AVE_DEN))
            loss_dsn3_ave.append(loss_dsn3_sum / float(AVE_DEN))
            loss_dsn4_ave.append(loss_dsn4_sum / float(AVE_DEN))
            loss_dsn5_ave.append(loss_dsn5_sum / float(AVE_DEN))
            loss_fuse_ave.append(loss_fuse_sum / float(AVE_DEN))

            count = 1
            loss_a_sum = 0
            loss_dsn1_sum = 0
            loss_dsn2_sum = 0
            loss_dsn3_sum = 0
            loss_dsn4_sum = 0
            loss_dsn5_sum = 0
            loss_fuse_sum = 0

    x_axis_len = len(iteration) / AVE_DEN
    x_axis = []
    for i in range(1, x_axis_len + 1):
        x_axis.append(i * AVE_DEN * STEP_SIZE)

    fig = plt.figure(figsize=(10, 15))
    plt.plot(x_axis, loss_a_ave, label='overall loss')
    plt.plot(x_axis, loss_dsn1_ave, label = 'dsn1 loss')
    plt.plot(x_axis, loss_dsn2_ave, label = 'dsn2 loss')
    plt.plot(x_axis, loss_dsn3_ave, label = 'dsn3 loss')
    plt.plot(x_axis, loss_dsn4_ave, label = 'dsn4 loss')
    plt.plot(x_axis, loss_dsn5_ave, label = 'dsn5 loss')
    plt.plot(x_axis, loss_fuse_ave, label = 'fuse loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('../../logs/loss.png')

    file.close()


if __name__ == '__main__':
    plot_ave_loss('../../logs/loss_all.txt')
