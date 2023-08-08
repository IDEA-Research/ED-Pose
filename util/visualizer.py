# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Author  :   Jie Yang
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms

from util.utils import renorm


class ColorMap():
    def __init__(self, basergb=[255, 255, 0]):
        self.basergb = np.array(basergb)

    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1)  # h, w, 3
        attn1 = attnmap.copy()[..., None]  # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


def rainbow_text(x, y, ls, lc, **kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    t = plt.gca().transData
    fig = plt.gcf()
    plt.show()

    # horizontal version
    for s, c in zip(ls, lc):
        text = plt.text(x, y, " " + s + " ", color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')



class COCOVisualizer():
    def __init__(self, coco=None, tokenlizer=None) -> None:
        self.coco = coco

    def visualize(self, img, tgt, caption=None, dpi=180, savedir='vis'):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        img = renorm(img).permute(1, 2, 0)
        fig=plt.figure(frameon=False)
        dpi = plt.gcf().dpi
        fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = plt.gca()

        ax.imshow(img, aspect='equal')

        self.addtgt(tgt)

        if caption is None:
            savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']),
                                             str(datetime.datetime.now()).replace(' ', '-'))
        else:
            savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']),
                                                str(datetime.datetime.now()).replace(' ', '-'))
        print("savename: {}".format(savename))
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=dpi)
        plt.close()

    def addtgt(self, tgt):
        """

        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        numbox = tgt['boxes'].shape[0]
        color_kpt = [[0.00, 0.00, 0.00],
                     [1.00, 1.00, 1.00],
                     [1.00, 0.00, 0.00],
                     [1.00, 1, 00., 0.00],
                     [0.50, 0.16, 0.16],
                     [0.00, 0.00, 1.00],
                     [0.69, 0.88, 0.90],
                     [0.00, 1.00, 0.00],
                     [0.63, 0.13, 0.94],
                     [0.82, 0.71, 0.55],
                     [1.00, 0.38, 0.00],
                     [0.53, 0.15, 0.34],
                     [1.00, 0.39, 0.28],
                     [1.00, 0.00, 1.00],
                     [0.04, 0.09, 0.27],
                     [0.20, 0.63, 0.79],
                     [0.94, 0.90, 0.55]]
        color = []
        color_box= 	[0.49,0.99,0]
        color_kpt_bbox = []
        polygons_kpt = []
        boxes_kpt = []
        polygons = []
        boxes = []
        for box in tgt['boxes'].cpu():
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                    [bbox_x + bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            color.append(color_box)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none',linestyle="--", edgecolors=color, linewidths=1.5)
        ax.add_collection(p)

        if 'strings_positive' in tgt:
            assert len(tgt['strings_positive']) == numbox, f"{len(tgt['strings_positive'])} = {numbox}, "
            for idx, strlist in enumerate(tgt['strings_positive']):
                cate_id = int(tgt['labels'][idx])
                _string = str(cate_id) + ':' + ' '.join(strlist)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)

        if 'attn' in tgt:
            if isinstance(tgt['attn'], tuple):
                tgt['attn'] = [tgt['attn']]
            for item in tgt['attn']:
                attn_map, basergb = item
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-3)
                attn_map = (attn_map * 255).astype(np.uint8)
                cm = ColorMap(basergb)
                heatmap = cm(attn_map)
                ax.imshow(heatmap)

        if 'keypoints' in tgt:
            sks = np.array(self.coco.loadCats(1)[0]['skeleton']) - 1
            for idx, ann in enumerate(tgt['keypoints']):
                if "kpt_bbox" in tgt:
                    for kpt_bbox in tgt['kpt_bbox'][idx].cpu():
                        unnormbbox = kpt_bbox * torch.Tensor([W, H, W, H])
                        unnormbbox[:2] -= unnormbbox[2:] / 2
                        [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
                        boxes_kpt.append([bbox_x, bbox_y, bbox_w, bbox_h])
                        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                                [bbox_x + bbox_w, bbox_y]]
                        np_poly = np.array(poly).reshape((4, 2))
                        polygons_kpt.append(Polygon(np_poly))

                        color_kpt_bbox.append(color_box)
                    p_kpt = PatchCollection(polygons_kpt, facecolor=color_kpt, linewidths=0, alpha=0.1)
                    ax.add_collection(p_kpt)
                    p_kpt = PatchCollection(polygons_kpt, facecolor='none', edgecolors=color_kpt, linewidths=1)
                    ax.add_collection(p_kpt)
                kp = np.array(ann.cpu())
                Z = kp[:34] * np.array([W, H] * 17)
                V = kp[34:]
                x = Z[0::2]
                y = Z[1::2]
                v = V
                if len(color) > 0:
                    c = color[idx % len(color)]
                else:
                    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        plt.plot(x[sk], y[sk], linewidth=2, color=c)

                for i in range(17):
                    c_kpt = color_kpt[i]
                    plt.plot(x[i], y[i], 'o', markersize=6, markerfacecolor=c_kpt, markeredgecolor='k', markeredgewidth=0.5)
        ax.set_axis_off()

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))

                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k',
                             markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])
