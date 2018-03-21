import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.test_img_names = ["./datasets/facades/test/2.jpg",
                            "./datasets/facades/test/8.jpg",
                            "./datasets/facades/test/9.jpg",
                            "./datasets/facades/test/10.jpg",
                            "./datasets/facades/test/12.jpg",
                            "./datasets/facades/test/15.jpg",
                            "./datasets/facades/test/20.jpg",
                            "./datasets/facades/test/26.jpg",
                            "./datasets/facades/test/42.jpg",
                            "./datasets/facades/test/44.jpg",
                            "./datasets/facades/test/51.jpg",
                            "./datasets/facades/test/59.jpg",
                            "./datasets/facades/test/77.jpg",
                            "./datasets/facades/test/87.jpg",
                            "./datasets/facades/test/90.jpg",
                            "./datasets/facades/test/96.jpg",
                            "./datasets/facades/test/101.jpg"]

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        if is_testing:
            batch_images = np.random.choice(self.test_img_names, size=batch_size)
        else:
            batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
