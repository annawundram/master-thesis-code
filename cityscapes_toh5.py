import torch
import torchvision
import numpy as np
from collections import namedtuple
import h5py
import random

# ---------- DISCLAIMER: from official cityscapes github repo ----------
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labelss = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      7 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      7 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      7 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      7 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      7 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      7 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      7 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        0 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,       0, 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      0 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        1 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        1 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        1 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      7 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      7 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      7 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        2 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      7 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        2 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        2 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        3 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        3 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       4 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       5 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       5 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       6 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       6 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       6 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      7 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      7 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       6 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       6 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       6 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      7 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
# --------------------------------------------------------------------------------

def write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, images,
                                labels):
    """ writes range of 4 images to hdf5 file
                    Parameters:
                    ---------------
                    counter_from   write from
                    counter_to     write to
                    images_data    hdf5 dataset
                    labels_data    hdf5 dataset
                    images         list of images as numpy arrays
                    labels         list of labels as numpy arrays
    """
    # add images
    images = np.asarray(images)
    images_data[counter_from:counter_to] = images

    # add labels
    labels = np.asarray(labels)
    labels_data[counter_from:counter_to] = labels


def add_images(images_data, labels_data, dataset, counter_from, idxs):
    """ preprocesses images and adds them to .
            Parameters:
            ---------------
            images_data        hdf5 dataset
            labels_data        hdf5 dataset
            diagnoses_data     hdf5 dataset
            dataset           torch dataset
            counter_from       int
    """
    max_write_buffer = 4
    write_buffer = 0
    images = []
    labels = []
    ids = []

    # go through all images, then preprocess them and write them to hdf5 files in batches of five
    for i in idxs:
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
            write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, images,
                                labels)
            # delete cash/lists
            images.clear()
            labels.clear()
            ids.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4

        # load PIL images
        image_pillow = dataset[i][0]
        segmentation_pillow = dataset[i][1]

        # to numpy
        image = np.asarray(image_pillow)
        segmentation = np.asarray(segmentation_pillow)

        image_pillow.close()
        segmentation_pillow.close()
        images.append(image)

        # id to train id -> https://github.com/fregu856/segmentation/blob/f051c6a918e74bf54b8d44ad1cc2c8103ee0b9d6/preprocess_data.py#L89-L91
        id_to_trainId = {label.id: label.trainId for label in labelss}
        id_to_trainId_map_func = np.vectorize(id_to_trainId.get)
        segmentation = id_to_trainId_map_func(segmentation)
        labels.append(segmentation)

        ids.append(i)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
        write_range_to_hdf5(counter_from, counter_to, images_data, labels_data, images,
                            labels)

# -----------------------------------------------------------------------------
# 1. ----------------------------- create H5 file -----------------------------
# -----------------------------------------------------------------------------

h5_file_path = "cityscapes_group.h5"
h5_file = h5py.File(h5_file_path, "w")
file_path = '/mnt/qb/baumgartner/rawdata/cityscapes'
transf = torchvision.transforms.CenterCrop(320)

train_dataset = torchvision.datasets.Cityscapes(file_path, split="train", mode='fine',
                     target_type='semantic', transform=transf, target_transform=transf)

val_dataset = torchvision.datasets.Cityscapes(file_path, split="val", mode='fine',
                     target_type='semantic', transform=transf, target_transform=transf)

# number of images in each set
number_train_images = len(train_dataset)
number_val_images = len(val_dataset)

# split train images into test and train
number_images_shuffled = list(range(number_train_images))
random.shuffle(number_images_shuffled)
idx_test, idx_train = np.split(np.asarray(number_images_shuffled), [int(len(number_images_shuffled) * 0.5)])

# split val images into cal and val
number_images_shuffled = list(range(number_val_images))
random.shuffle(number_images_shuffled)
idx_cal, idx_val = np.split(np.asarray(number_images_shuffled), [int(len(number_images_shuffled) * 0.5)])

# get h5 datasets for train and val
img_data_train = h5_file.create_dataset("train/images", shape=(len(idx_train), 320, 320, 3), dtype="uint8")
img_data_val = h5_file.create_dataset("val/images", shape=(len(idx_val), 320, 320, 3), dtype="uint8")

label_data_train = h5_file.create_dataset("train/label", shape=(len(idx_train), 320, 320), dtype="uint8")
label_data_val = h5_file.create_dataset("val/label", shape=(len(idx_val), 320, 320), dtype="uint8")

# ---- train ----
add_images(img_data_train, label_data_train, train_dataset, 0, idx_train)

# ---- val ----
add_images(img_data_val, label_data_val, val_dataset, 0, idx_val)

# --------------------------------------------------------------------------
# get h5 datasets for test and cal
img_data_test = h5_file.create_dataset("test/images", shape=(len(idx_test), 320, 320, 3), dtype="uint8")
img_data_cal = h5_file.create_dataset("cal/images", shape=(len(idx_cal), 320, 320, 3), dtype="uint8")

label_data_test = h5_file.create_dataset("test/label", shape=(len(idx_test), 320, 320), dtype="uint8")
label_data_cal = h5_file.create_dataset("cal/label", shape=(len(idx_cal), 320, 320), dtype="uint8")

# ---- test ----
add_images(img_data_test, label_data_test, train_dataset, 0, idx_test)

# ---- cal ----
add_images(img_data_cal, label_data_cal, val_dataset, 0, idx_cal)

# --------------------------------------------------------------------------
h5_file.close()
