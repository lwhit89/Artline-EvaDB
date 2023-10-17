from abc import ABC

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
import fastai
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import open_image, load_learner, image, torch
import numpy as np
import urllib.request
import torchvision.transforms as T
from PIL import Image, ImageOps


class Artline(AbstractFunction):
    @property
    def name(self) -> str:
        return "yolo"

    @setup(cacheable=True, function_type="image_transformation", batchable=True)
    def setup(self):
        MODEL_URL = "https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1"
        urllib.request.urlretrieve(MODEL_URL, "ArtLine_650.pkl")
        path = Path(".")
        self.model = load_learner(path, "ArtLine_650.pkl")

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["image_path"],
                column_types=[NdArrayType.STR]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["result_image_path"],
                column_types=[
                    NdArrayType.STR,
                ]
            )
        ],
    )
    def forward(self, path_df):
        content = path_df[path_df.columns[0]]
        results = []
        for image_path in content:
            dot = image_path.find('.')
            result_path = image_path[:dot] + "_result" + image_path[dot:]

            img = PIL.Image.open(image_path).convert("RGB")
            img_t = T.ToTensor()(img)
            transform = T.ToPILImage()
            img_fast = fastai.vision.Image(img_t)

            p, img_hr, b = self.model.predict(img_fast)
            result = transform(img_hr)
            inv_result = PIL.ImageOps.invert(result)
            inv_result.save(result_path)
            results.append(result_path)
        return pd.DataFrame({"result_image_path": results})


# class FeatureLoss(nn.Module):
#     def __init__(self, m_feat, layer_ids, layer_wgts):
#         super().__init__()
#         self.m_feat = m_feat
#         self.loss_features = [self.m_feat[i] for i in layer_ids]
#         self.hooks = hook_outputs(self.loss_features, detach=False)
#         self.wgts = layer_wgts
#         self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
#                                            ] + [f'gram_{i}' for i in range(len(layer_ids))]
#
#     def make_features(self, x, clone=False):
#         self.m_feat(x)
#         return [(o.clone() if clone else o) for o in self.hooks.stored]
#
#     def forward(self, input, target):
#         out_feat = self.make_features(target, clone=True)
#         in_feat = self.make_features(input)
#         self.feat_losses = [base_loss(input, target)]
#         self.feat_losses += [base_loss(f_in, f_out) * w
#                              for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
#         self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
#                              for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
#         self.metrics = dict(zip(self.metric_names, self.feat_losses))
#         return sum(self.feat_losses)
#
#     def __del__(self): self.hooks.remove()