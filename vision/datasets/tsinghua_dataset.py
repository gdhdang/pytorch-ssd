import pathlib
import json
import logging
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

class TsinghuaDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            image_sets_file = self.root / "test/ids.txt"
        else:
            image_sets_file = self.root / "train/ids.txt"
        self.ids = TsinghuaDataset._read_image_ids(image_sets_file)

        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "tsinghua_labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("Tsinghua Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND', 
            'i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 
            'p11', 'p12', 'p19', 'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 
            'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 
            'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 
            'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"annotations.json"
        annos = json.loads(open(annotation_file).read())
        img = annos["imgs"][image_id]
        all_objs = img["objects"]

        boxes = []
        labels = []
        for obj in all_objs:
            class_name = obj["category"]
            # we are only concerned with classes in our list
            if class_name in self.class_dict:
                bbox = obj["bbox"]

                x1 = float(bbox["xmin"])
                y1 = float(bbox["ymin"])
                x2 = float(bbox["xmax"])
                y2 = float(bbox["ymax"])
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        annotation_file = self.root / f"annotations.json"
        annos = json.loads(open(annotation_file).read())
        img = annos["imgs"][image_id]

        image_file = self.root / img['path']
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image