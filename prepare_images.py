import urllib.request
import cv2
import numpy as np
import os


img_w = 300
img_h = 200

# Download image from imagenet
def store_raw_images(images_link,path):
    image_urls = urllib.request.urlopen(images_link).read().decode()
    pic_num = 1

    if not os.path.exists(path):
        os.makedirs(path)

    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, path + '/' + str(pic_num) + ".jpg")
            img = cv2.imread(path + '/' + str(pic_num) + ".jpg")
            resized_image = cv2.resize(img, (img_w, img_h))
            cv2.imwrite(path + '/' + str(pic_num) + ".jpg", resized_image)
            pic_num += 1
        except Exception as e:
            print(str(e))
    print(pic_num)


# Delete unsuccessful downloaded pictures
def find_uglies(path):
    match = False
    for file_type in [path]:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


def main():
    # guitars
    images_link_A = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02676566'
    # cellos
    images_link_B = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02992211'
    path_A = 'object_A'
    path_B = 'object_B'
    store_raw_images(images_link_A, path_A)
    store_raw_images(images_link_B, path_B)
    #find_uglies(path_A)
    #find_uglies(path_B)


if __name__ == '__main__':
    main()
