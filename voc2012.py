import xml.etree.cElementTree as ElementTree
from torch.utils.data.dataset import Dataset
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''

    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})

    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        raise AttributeError(f'No such Thing: {item}, in me : {self}')


class VOC:
    def __init__(self, split, root=None):
        assert split in ['train', 'val'], f'split name {split} not recognized!'
        if root is None:
            root = '/data/home/d/data/voc2012/VOCdevkit/VOC2012'
        self.root = root
        self.split = split
        files = open(pjoin(root, 'ImageSets/Main', split + '.txt'), 'r').read().splitlines()
        self.annotations = list(
            map(lambda x: pjoin(self.root, 'Annotations', x + '.xml'),
                files))
        self.images = list(map(
            lambda x: pjoin(self.root, 'JPEGImages', x.split('/')[-1].split('.')[0] + '.jpg'),
            self.annotations
        ))

    def __getitem__(self, item):
        item = int(item)
        img = np.array(Image.open(self.images[item]))
        assert img.shape.__len__() == 3
        anno = XmlDictConfig(ElementTree.parse(self.annotations[item]).getroot())
        anno = anno.object.bndbox
        cx = (anno.xmax + anno.xmin) // 2
        cy = (anno.ymax + anno.ymin) // 2
        width = anno.xmax - anno.xmin
        height = anno.ymax - anno.ymin

        return img, (cx, cy, width, height)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    voc = VOC('val')
    for i in voc.images:
        assert os.path.exists(i), f'{i} not exist!'
    for i in voc.annotations:
        assert os.path.exists(i), f'{i} not exist!'

    for img, anno in tqdm(voc):
        print(anno)
