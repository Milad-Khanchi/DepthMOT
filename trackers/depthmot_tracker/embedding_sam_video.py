from collections import OrderedDict
from pathlib import Path
import os
import pickle
import timm
import torch
import cv2
import torchvision
import torchreid
import numpy as np
from torch import nn
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Load .env file
load_dotenv()

# Get SAM2 path from environment
sam2_path = os.getenv("SAM2_PATH")
if sam2_path:
    sys.path.append(sam2_path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2_video_predictor
else:
    raise EnvironmentError("SAM2_PATH is not set in .env file")


class EmbeddingComputer:
    def __init__(self, dataset, test_dataset, grid_off, max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (224, 224)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch

        # Only used for the general ReID model (not FastReID)
        self.normalize = False

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            # Faulty Patch Correction
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ### TODO - Write a generalized split logic
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        # breakpoint()
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2], :]

                if viz:  ## TODO - change it from torch tensor to numpy array
                    dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255,
                    )
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                # print("test ", patch.shape)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, (256, 128))
                patches.append(patch)

        patches = torch.cat(patches, dim=0)

        # print("Patches shape ", patches.shape)
        # patches = np.array(patches)
        # print("ALL SPLIT PATCHES SHAPE - ", patches.shape)

        return patches

    def compute_embedding(self, img, bbox, tag, img_path):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            self.initialize_model()

        # Generate all of the patches
        crops = []
        if self.grid_off:
            # Basic embeddings
            h, w = img.shape[:2]
            results = np.round(bbox).astype(np.int32)
            results[:, 0] = results[:, 0].clip(0, w)
            results[:, 1] = results[:, 1].clip(0, h)
            results[:, 2] = results[:, 2].clip(0, w)
            results[:, 3] = results[:, 3].clip(0, h)

            crops = []
            for p in results:
                crop = img[p[1] : p[3], p[0] : p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                #crop = torch.as_tensor(crop.transpose(2, 0, 1))
                #crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # Grid patch embeddings
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)
        #crops = torch.cat(crops, dim=0)
        # Create embeddings and l2 normalize them
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx : idx + self.max_batch]
            #batch_crops = batch_crops.cuda()
            with torch.no_grad():
                iner = [img_path[0][0], img_path[1][0]]
                inference_state = self.model.init_state(video_path=iner)
                self.model.reset_state(inference_state)
                for i in range(len(batch_crops)):
                    _, out_obj_ids, out_mask_logits = self.model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=i,
                        box=bbox[i],
                    )
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        transform_output = transforms.Compose([
            transforms.Resize((256, 256)),
        ])
        embs0 = [transform_output(element).squeeze(0).flatten(start_dim=0) for element in video_segments[0].values()]
        embs0 = torch.stack(embs0)
        embs0 = torch.nn.functional.normalize(embs0, dim=-1)
        embs1 = [transform_output(element).squeeze(0).flatten(start_dim=0) for element in video_segments[1].values()]
        embs1 = torch.stack(embs1)
        embs1 = torch.nn.functional.normalize(embs1, dim=-1)
        embs = (embs0.cpu().numpy(), embs1.cpu().numpy())

        #here
        # plt.close("all")
        # a = np.random.random(1)
        # for out_frame_idx in range(0, len(iner)):
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(Image.open(iner[out_frame_idx]))
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        #
        #     plt.savefig(f'./cache/sam_img/{a}{out_frame_idx}.png')

        if not self.grid_off:
            pass
            #embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        #embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        if self.dataset == "mot17":
            if self.test_dataset:
                path = "external/weights/mot17_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = "external/weights/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "dance":
            path = "external/weights/dance_sbs_S50.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        # model = FastReID(path)
        # model.eval()
        # model.cuda()
        # model.half()
        # self.model = model
        # Load a pretrained Vision Transformer model

        sam2_checkpoint = os.getenv("SAM2_CHECKPOINT")
        model_cfg = os.getenv("MODEL_CFG")


        self.model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')


    def _get_general_model(self):
        """Used for the half-val for MOT17/20.

        The MOT17/20 SBS models are trained over the half-val we
        evaluate on as well. Instead we use a different model for
        validation.
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True

    def dump_cache(self):
        if self.cache_name:
            # with open(self.cache_path.format(self.cache_name), "wb") as fp:
            #     pickle.dump(self.cache, fp)
            pass

def show_mask(mask, ax, obj_id=None, random_color=False):
    mask = (mask > 0.0).cpu().numpy()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))