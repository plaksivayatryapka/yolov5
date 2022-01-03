import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

import cv2
import albumentations as alb


parser = ArgumentParser()


class PairImgLabel:
    """
    create img-label pair by label txt file
    """

    def __init__(self, path_dataset, filename_txt, image_extension):
        self.path_dataset = path_dataset
        self.filename_txt = filename_txt

        self.filename_base = Path(filename_txt).stem
        self.image_extension = image_extension
        self.filename_img = f'{self.filename_base}.{self.image_extension}'

        self.filepath_txt = os.path.join(self.path_dataset, self.filename_txt)
        self.filepath_img = os.path.join(self.path_dataset, self.filename_img)

        self.classes, self.bboxes = self.get_labels_data()
        self.type_ = None  # train or test

    def set_type(self, type_):
        self.type_ = type_

    def get_labels_data(self):
        bboxes = list()
        classes = list()

        with open(self.filepath_txt, 'r') as f:
            for line in f:
                lst = line.split()
                class_ = int(lst[0])
                bbox = lst[1:]
                bbox = [float(i) for i in bbox]

                classes.append(class_)
                bboxes.append(bbox)

        return classes, bboxes

    def move(self):
        # move image file
        filepath_from = os.path.join(self.path_dataset, self.filename_img)
        filepath_to = os.path.join(self.path_dataset, 'images', self.type_, self.filename_img)
        shutil.move(filepath_from, filepath_to)
        self.filepath_img = filepath_to

        # move text file
        filepath_from = os.path.join(self.path_dataset, self.filename_txt)
        filepath_to = os.path.join(self.path_dataset, 'labels', self.type_, self.filename_txt)
        shutil.move(filepath_from, filepath_to)
        self.filepath_txt = filepath_to


class Augmentation:
    def __init__(self, pair):
        self.pair = pair
        self.value = None

    def save_label(self, filepath_transformed_label, classes, bboxes):
        with open(filepath_transformed_label, 'w') as f:
            for class_, bbox in zip(classes, bboxes):
                bbox_string = ' '.join(bbox)
                label_string = f'{class_} {bbox_string}\n'
                f.write(label_string)

    def create_rotation(self, value):
        rotation = alb.Affine(rotate=value,
                              always_apply=True,
                              p=1.0)

        return rotation

    def create_rgbshift(self, value):
        rotation = alb.RGBShift(r_shift_limit=value,
                                g_shift_limit=value,
                                b_shift_limit=value,
                                always_apply=True,
                                p=1.0)

        return rotation

    def create_jpegcompression(self, value):
        rotation = alb.JpegCompression(quality_lower=value,
                                       quality_upper=value + 1,
                                       always_apply=True,
                                       p=1.0)

        return rotation

    def create_transformation(self, transformation, transformation_name):
        if transformation_name == 'rgb_shift':
            transformation2 = self.create_rotation(90)
            transformations = [transformation, transformation2]

        # elif transformation_name == 'rgb_shift2':
        #     transformation2 = self.create_rotation(270)
        #     transformations = [transformation, transformation2]

        else:
            transformations = [transformation]

        transform_rotation = alb.Compose(transformations,
                                         bbox_params=alb.BboxParams(format='yolo',
                                                                    label_fields=['class_labels']))
        return transform_rotation

    def apply_transformation(self, transformation_name, value):
        if transformation_name == 'rotate':
            transformation = self.create_rotation(value)

        if transformation_name == 'rgb_shift' or transformation_name == 'rgb_shift2':
            transformation = self.create_rgbshift(value)

        if transformation_name == 'jpeg_compression':
            transformation = self.create_jpegcompression(value)

        transform = self.create_transformation(transformation, transformation_name)
        image = cv2.imread(self.pair.filepath_img)

        transformed = transform(image=image, bboxes=self.pair.bboxes, class_labels=self.pair.classes)
        image = transformed['image']
        bboxes = transformed['bboxes']
        classes = transformed['class_labels']

        bboxes = [[str(j) for j in i] for i in bboxes]

        filename_transformed_image = Path(self.pair.filename_img).stem + f'_{transformation_name}_{value}.jpg'
        filename_transformed_label = Path(self.pair.filepath_txt).stem + f'_{transformation_name}_{value}.txt'

        filepath_transformed_image = os.path.join(self.pair.path_dataset, 'images', self.pair.type_, filename_transformed_image)
        filepath_transformed_label = os.path.join(self.pair.path_dataset, 'labels', self.pair.type_, filename_transformed_label)

        self.save_label(filepath_transformed_label, classes, bboxes)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filepath_transformed_image, image)

        path = os.path.join(self.pair.path_dataset, 'labels', self.pair.type_)
        pair = PairImgLabel(path, filename_transformed_label, 'jpg')
        pair.set_type = self.pair.type_
        return pair


class Dataset:
    def __init__(self, url, dir_datasets, dataset_name, image_extension, test_ratio, augment_transformations, augment_values, download=True):
        self.url = url
        self.url_filename = Path(self.url).name
        self.dir_datasets = dir_datasets
        self.dir_dataset = dataset_name
        self.path_dataset = os.path.join(self.dir_datasets, self.dir_dataset)
        self.test_ratio = test_ratio
        self.image_extension = image_extension
        self.augment_transformations = augment_transformations
        self.augment_values = augment_values
        self.to_download = download
        self.pairs = None

    def download(self):
        urllib.request.urlretrieve(self.url, self.url_filename)  # save as self.url_filename

    def unzip(self):
        Path(self.path_dataset).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.url_filename, 'r') as zip_ref:
            zip_ref.extractall(self.path_dataset)

    def create_pairs(self):
        pairs = list()
        for filename in os.listdir(self.path_dataset):
            if not filename.endswith('.txt'):  # create img-label pairs by label file
                continue

            pair = PairImgLabel(self.path_dataset, filename, self.image_extension)
            pairs.append(pair)

        return pairs

    def get_train_test_split(self):
        train_pairs, test_pairs = train_test_split(self.pairs, test_size=self.test_ratio, random_state=42)
        for pair in self.pairs:
            if pair in test_pairs:
                pair.set_type('val')
            else:
                pair.set_type('train')

    def create_train_test_dirs(self):
        for i in ['images', 'labels']:
            for j in ['train', 'val']:
                Path(os.path.join(self.path_dataset, i, j)).mkdir(parents=True, exist_ok=True)

    def move_train_test(self):
        for pair in self.pairs:
            pair.move()

    def augmentate_pair(self, pair):
        a = Augmentation(pair)

        for augment_transformation, value in zip(self.augment_transformations, self.augment_values):
            pair = a.apply_transformation(augment_transformation, value)

        return pair

    def augment(self):
        augmented_pairs = list()
        for pair in self.pairs:
            pair_new = self.augmentate_pair(pair)
            augmented_pairs.append(pair_new)

        self.pairs += augmented_pairs

    def create(self):
        if self.to_download:
            self.download()
            self.unzip()

        self.pairs = self.create_pairs()
        self.get_train_test_split()
        self.create_train_test_dirs()
        self.move_train_test()
        self.augment()


parser.add_argument("--url")
parser.add_argument("--dir_datasets")
parser.add_argument("--dataset_name")
parser.add_argument("--image_extension")
parser.add_argument("--test_ratio", type=float)
parser.add_argument("--augment_transformations", nargs='+')
parser.add_argument("--augment_values", nargs='+', type=int)
parser.add_argument("--to_download", type=int)

args = parser.parse_args()

url = args.url
dir_datasets = args.dir_datasets
dataset_name = args.dataset_name
image_extension = args.image_extension
test_ratio = args.test_ratio
augment_transformations = args.augment_transformations
augment_values = args.augment_values
to_download = args.to_download

if to_download == 1:
    to_download = True
else:
    to_download = False

d = Dataset(url, dir_datasets, dataset_name, image_extension, test_ratio, augment_transformations, augment_values, download=to_download)
d.create()
