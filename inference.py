#import PyQt5

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2

# from detectron2.data import MetadataCatalog
# from detectron2.structures import Instances
# from detectron2.utils.visualizer import Visualizer, VisImage
# import numpy as np


# Salmon outline detector
#
# This program detects the outline of a swimming salmon from experiments as part of the Master's thesis of
# Eirik Ruben Grimholt Søvik at Marine Technological Center, NTNU, Trondheim.
#
# Relevant variables:
#       - OUTPUT_FOLDER
#       -


#import argparse

import numpy as np
#import re

#from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
#from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import  VisImage #,Visualizer

import math
#from detectron2.engine import DefaultTrainer
#from detectron2 import model_zoo
from PIL import Image
import torch #, torchvision

print(torch.__version__, torch.cuda.is_available())


from detectron2.utils.logger import setup_logger
#import os, cv2, random
import cv2
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") # compatibility with PyQt5
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
#from detectron2.data import DatasetCatalog

#from detectron2.data.datasets import register_coco_instances
#from detectron2.config import LazyConfig
#from detectron2.engine import DefaultTrainer
#from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import modeling

setup_logger()

import time, os #, fnmatch, shutil


def calculateDistance(x1, y1, x2, y2):
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist

def get_cropped_leaf(img, predictor, return_mapping=False, resize=None):
    # convert to numpy
    img = np.array(img)[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # get prediction
    outputs = predictor(img)

    # get boxes
    ins = outputs["instances"]
    pred_masks = ins.get_fields()["pred_masks"]
    boxes = ins.get_fields()["pred_boxes"]

    # get main leaf mask if the area is >= the mean area of boxes and is closes to the centre
    masker = pred_masks[np.argmin(
        [calculateDistance(x[0], x[1], int(img.shape[1] / 2), int(img.shape[0] / 2)) for i, x in
         enumerate(boxes.get_centers()) if (boxes[i].area() >= torch.mean(boxes.area()).to("cpu")).item()])].to(
        "cpu").numpy().astype(np.uint8)

    # mask image
    mask_out = cv2.bitwise_and(img, img, mask=masker)

    # find contours and boxes
    contours, hierarchy = cv2.findContours(masker.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]
    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    # crop image
    cropped = get_cropped(rotrect, box, mask_out)

    # resize
    rotated = MakeLandscape()(Image.fromarray(cropped))

    if not resize == None:
        resized = ResizeMe((resize[0], resize[0], resize[1]))(rotated)
    else:
        resized = rotated

    if return_mapping:
        img = cv2.drawContours(img, [box], 0, (0, 0, 255), 10)
        img = cv2.drawContours(img, contours, -1, (255, 150,), 10)
        return resized, ResizeMe((int(resize[0]), int(resize[1])))(Image.fromarray(img))

    return resized, mask_out, contours, contour

def get_cropped(rotrect, box, image):
    width = int(rotrect[1][0])
    height = int(rotrect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


# image manipulations

class ResizeMe(object):
    # resize and center image in desired size
    def __init__(self, desired_size):

        self.desired_size = desired_size

    def __call__(self, img):

        img = np.array(img).astype(np.uint8)

        desired_ratio = self.desired_size[1] / self.desired_size[0]
        actual_ratio = img.shape[0] / img.shape[1]

        desired_ratio1 = self.desired_size[0] / self.desired_size[1]
        actual_ratio1 = img.shape[1] / img.shape[0]

        if desired_ratio < actual_ratio:
            img = cv2.resize(img, (int(self.desired_size[1] * actual_ratio1), self.desired_size[1]), None,
                             interpolation=cv2.INTER_AREA)
        elif desired_ratio > actual_ratio:
            img = cv2.resize(img, (self.desired_size[0], int(self.desired_size[0] * actual_ratio)), None,
                             interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (self.desired_size[0], self.desired_size[1]), None, interpolation=cv2.INTER_AREA)

        h, w, _ = img.shape

        new_img = np.zeros((self.desired_size[1], self.desired_size[0], 3))

        hh, ww, _ = new_img.shape

        yoff = int((hh - h) / 2)
        xoff = int((ww - w) / 2)

        new_img[yoff:yoff + h, xoff:xoff + w, :] = img

        return Image.fromarray(new_img.astype(np.uint8))


class MakeLandscape():
    # flip if needed
    def __init__(self):
        pass

    def __call__(self, img):
        if img.height > img.width:
            img = np.rot90(np.array(img))
            img = Image.fromarray(img)
        return img

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off')
    plt.show() #maybe turn off?
    print("cv2_imshow function used!")


def load_dataset(DATASETNAME):
    # MODELNAME = "/" + DATASETNAME + "_model.pth"

    DATASETNAME = "inputs/salmon1015to1020"

    print("Setting config...")
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("salmon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8  # previously 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 400  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    # DefaultTrainer.auto_scale_workers(cfg,8)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:

    cfg.MODEL.WEIGHTS = os.path.join("output/Models/30des2021model.pth")  # path to model previously trained
    print(cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    print("Loading default predictor with custom config")
    predictor = DefaultPredictor(cfg)
    model = modeling.build_model(cfg)

    DetectionCheckpointer(model).load(cfg.OUTPUT_DIR + "/model_final.pth")

    print("Testing to see if we got anything useful. Let us load our new dataset and print some of the")
    print("predictions to see how they are!")
    #data = DatasetCatalog.get("salmon_train")

    #return data, cfg, model, predictor
    return cfg, model, predictor

from scipy.interpolate import UnivariateSpline

def smoother(mask, plot=False, k=1, s=2, N=400):
    x = mask[:, 0]
    y = mask[:, 1]

    x_new = np.zeros([len(x)+1])
    y_new = np.zeros([len(y)+1])

    x_new[0:len(x)] = x
    x_new[-1] = x[0]
    y_new[0:len(y)] = y
    y_new[-1] = y[0]

    x = x_new
    y = y_new

    points = np.vstack((x, y)).T

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # for s in range(25,50,5):
    # Build a list of the spline function, one for each dimension:
    # k = 1
    # s = 50
    splines = [UnivariateSpline(distance, coords, k=k, s=s) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, N)
    points_fitted = np.vstack(spl(alpha) for spl in splines).T

    # for spl in splines:
    #    points_fitted = np.vstack(spl(alpha)).T

    if plot == True:
        # Graph:
        # plt.plot(*points.T, 'ok', label='original points')
        # plt.plot(mask[:, 0], mask[:, 1])
        plt.plot(points_fitted[:, 0], points_fitted[:, 1], 'ro', label='fitted spline k=' + str(k) + ', s=' + str(s))
        plt.plot(x,y,'bx')
        plt.plot(x[0],y[0],'go',x[-1],y[-1],'go')
        plt.axis('equal')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.show()
        # time.sleep(1)
        # plt.pause(10)

        plt.show()
        wait = input('Press any key to go to the next iteration')
    return points_fitted


# import mask_smoother



if __name__ == "__main__":

    save = False
    plotting = True



    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    OUTPUT_FOLDER = ("mask_output_" + timestamp)
    # DATASETNAME = "30des2021coco"
    DATASETNAME = "inputs/salmon1015to1020"

    OUTPUT_FOLDER = ("mask_output_april_" + timestamp)
    # DATASETNAME = "inputs/salmon1015to1020"
    cfg, model, predictor = load_dataset(DATASETNAME)

    print("Loading dataset: " + DATASETNAME)
    print("Output folder: " + OUTPUT_FOLDER)
    print("Saving set to: " + str(save))

    count = 1
    dirlist = os.listdir(DATASETNAME)
    print("Sequencing images now. In total " + str(len(dirlist)) + " images")
    image_file: str
    if save==True:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            os.makedirs(OUTPUT_FOLDER + "/images")
            os.makedirs(OUTPUT_FOLDER + "/masks")

    for image_file in dirlist:


        print("Processing image " + str(count) + " of " + str(len(dirlist)))
        count += 1
        # image_file = "salmon_5_1_extract.jpg"
        # img: np.ndarray = cv2.imread(image_file)

        img: np.ndarray = cv2.imread(DATASETNAME + "/" + image_file)
        try:
            output: Instances = predictor(img)["instances"]
        except AttributeError: # skip image if there is an error
            print("Image skipped! Image " + image_file + " encountered an attribute error and had to be skipped")
            continue


        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                       scale=1.0)
        result: VisImage = v.draw_instance_predictions(output.to("cpu"))
        result_image: np.ndarray = result.get_image()[:, :, ::-1]
        cv2_imshow(result_image)

        # get file name without extension, -1 to remove "." at the end
        # out_file_name: str = re.search(r"(.*)\.", image_file).group(0)[:-1]
        # out_file_name += "_processed.png"

        resized, mask_out, contours, contour = get_cropped_leaf(img, predictor)  # returntype true

        data_in_array = np.array(contour)
        transposed = data_in_array.T
        x, y = transposed

        contour_2D = np.vstack((x, y)).T
        #mask, plot = False, k = 1, s = 2, N = 400
        smooth_contour_2D = smoother(contour_2D, plot=plotting, k=1, s=3, N=400)

        #plotting=False
        if plotting:
            fig, ax = plt.subplots()

            ax.plot(contour_2D[:, 0], contour_2D[:, 1], 'o')
            #ax.plot(smooth_contour_2D[:, 0], smooth_contour_2D[:, 1])
            ax.legend(["Contour", "Smoothed countour"])
            ax.set_title("Contour generated by detectron2")
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            plt.show()

        # fig, ax = plt.subplots(1,1)
        # ax.plot(contour_3D[:, 0], contour_3D[:, 1], 'ro')
        # ax.axis('equal')
        # plt.title("contour_3D")
        # fig.show()

        # plt.plot(x_val, y_val)
        # plt.show()

        # save numpy array as npy file
        from numpy import asarray
        from pathlib import Path
        if save==True:

            #cv2.imwrite(OUTPUT_FOLDER+"/images/"+image_file, result_image)

            path_filename = Path(image_file)
            image_file_nosuffix = str(path_filename.with_suffix(''))

            #os.makedirs(OUTPUT_FOLDER + "/masks/" + image_file )#os.path.basename(image_file["file_name"]))
            os.makedirs(OUTPUT_FOLDER + "/masks/" + image_file_nosuffix)
            #cv2.imwrite(OUTPUT_FOLDER + "/images/" + os.path.basename(image_file["file_name"]) + ".jpg", result_image)


            np.save(OUTPUT_FOLDER + "/masks/" + image_file_nosuffix + "/" + image_file_nosuffix, smooth_contour_2D)

            print("Masks saved!")


    print("script finished!")
