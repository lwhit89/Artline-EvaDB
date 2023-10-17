from pathlib import Path

import evadb
import os

import fastai
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
from io import BytesIO
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image, ImageOps


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()




# image_folder = os.listdir("./test_images")
# print(image_folder)
# for i in range(len(image_folder)):
#     path_to_image = os.path.join("C:\\\\Users\\\\whitl\\\\PycharmProjects\\\\Artline-EvaDB\\\\test_images\\\\", image_folder[i])
#     print(f"({i}, \'{path_to_image}\'),")

cursor = evadb.connect().cursor()

print(cursor.query("SHOW FUNCTIONS;").df())
cursor.query("""DROP DATABASE mysql_data""")
cursor.query("""CREATE DATABASE mysql_data WITH ENGINE = 'mysql', PARAMETERS = {
     "user": "root",
     "password": "",
     "host": "localhost",
     "port": "3306",
     "database": "evadb"
};
""").df()

cursor.query("CREATE FUNCTION IF NOT EXISTS Artline IMPL 'evadb_data/functions/artline_image_transformer.py'").df()
print(cursor.query("SELECT Artline(image_path) FROM mysql_data.images WHERE image_id < 5").df())
cursor.query("DROP FUNCTION IF EXISTS Artline").df()
