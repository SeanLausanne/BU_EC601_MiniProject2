import numpy as np
import os
import cv2

img_w = 100
img_h = 100
train_path_A = os.getcwd() + '/guitars_cellos/train/guitars/'
train_path_B = os.getcwd() + '/guitars_cellos/train/cellos/'
train_path_resized = os.getcwd() + '/guitars_cellos/train/resized/'
test_path_A = os.getcwd() + '/guitars_cellos/test/guitars/'
test_path_B = os.getcwd() + '/guitars_cellos/test/cellos/'
test_path_resized = os.getcwd() + '/guitars_cellos/test/resized/'


def get_data(path_A, path_B, path_resized):
    image_path_list = []
    label = []
    resized_path_list = []

    for filename in os.listdir(path_A):
        image_path_list.append(path_A + filename)
        label.append(0)

    for filename in os.listdir(path_B):
        image_path_list.append(path_B + filename)
        label.append(1)

    img_counter = 0
    for i in image_path_list:
        print(img_counter)
        img = cv2.imread(i)
        resized_img = cv2.resize(img, (img_w, img_h))
        temp_path = path_resized + str(img_counter) + ".jpg"
        cv2.imwrite(temp_path, resized_img)
        resized_path_list.append(temp_path)
        img_counter += 1

    all_data = np.array([resized_path_list, label])
    all_data = all_data.transpose()
    np.random.shuffle(all_data)

    return all_data


def process_data(data):
    img_matrix = []
    for i in data:
        img = cv2.imread(i[0])
        img = img.astype(float)
        temp = img.reshape((img_w*img_h,3))
        img_reshape = np.hstack((temp[:, 2], temp[:, 1], temp[:, 0]))
        img_matrix.append(img_reshape)

    img_matrix = np.asarray(img_matrix)
    return img_matrix


def prepare_data():
    train_list = get_data(train_path_A, train_path_B, train_path_resized)
    test_list = get_data(test_path_A, test_path_B, test_path_resized)

    train_label = train_list[:, 1]
    test_label = test_list[:, 1]
    train_label = train_label.astype(int)
    test_label = test_label.astype(int)

    train_data = process_data(train_list)
    test_data = process_data(test_list)

    mean_image = np.mean(train_data, axis=0)
    train_data -= mean_image
    test_data -= mean_image

    classes = ['objectA', 'objectB']

    data_dict = {
        'images_train': train_data,
        'labels_train': train_label,
        'images_test': test_data,
        'labels_test': test_label,
        'classes': classes
    }
    return data_dict


def main():
    prepare_data()


if __name__ == '__main__':
    main()




