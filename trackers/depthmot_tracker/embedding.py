from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np
from scipy.stats import gaussian_kde
from external.adaptors.fastreid_adaptor import FastReID

import sys
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Get SAM2 path from environment
sam2_path = os.getenv("SAM2_PATH")
if sam2_path:
    sys.path.append(sam2_path)
    from sam2.build_sam import build_sam2_video_predictor
else:
    raise EnvironmentError("SAM2_PATH is not set in .env file")
import depth_pro

class EmbeddingComputer:
    def __init__(self, dataset, test_dataset, grid_off, max_batch=1024):
        self.model = None
        self.model_mask = None
        self.model_depth = None
        self.model_depth_transf = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)
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
        bbox.astype(np.int32)
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
            embs, mask_embs, normalized_depths = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs, (mask_embs[0].numpy(), mask_embs[1].numpy()), normalized_depths

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
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # Grid patch embeddings
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)
        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx : idx + self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        with torch.no_grad():

            iner = [img_path[0][0], img_path[1][0]]

            # SAM 2

            inference_state = self.model_mask.init_state(video_path=iner)
            self.model_mask.reset_state(inference_state)
            for i in range(len(results)):
                _, out_obj_ids, out_mask_logits = self.model_mask.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i,
                    box=results[i],
                )
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.model_mask.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            transform_output = transforms.Compose([
                transforms.Resize((256, 256)),
            ])
            embs0 = [(element).squeeze(0) for element in
                     video_segments[0].values()]
            embs0 = torch.stack(embs0)
            # embs0 = torch.nn.functional.normalize(embs0, dim=-1)
            embs1 = [(element).squeeze(0) for element in
                     video_segments[1].values()]
            embs1 = torch.stack(embs1)
            # embs1 = torch.nn.functional.normalize(embs1, dim=-1)

            #mask_embs = (embs0.cpu().numpy(), embs1.cpu().numpy())
            mask_embs = ((embs0 > 0).cpu().to(torch.bool), (embs1 > 0).cpu().to(torch.bool))

            # #plot masks
            # plt.close("all")
            # a = np.random.random(1)
            # for out_frame_idx in range(0, len(iner)):
            #     plt.figure(figsize=(6, 4))
            #     plt.title(f"frame {out_frame_idx}")
            #     plt.imshow(Image.open(iner[out_frame_idx]))
            #     if out_frame_idx == 0:
            #         for bxr in range(bbox.shape[0]):
            #             show_box(bbox[bxr], plt.gca())
            #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            #
            #     plt.savefig(f'/cache/sam_img/{a}{out_frame_idx}.png')
            # #until here

            # depth

            # Load and preprocess an image.
            image, _, f_px = depth_pro.load_rgb(iner[0])
            img = self.model_depth_transf(image)

            # Generate depth maps
            prediction = self.model_depth.infer(img, f_px=f_px)
            depth = prediction["depth"]  # Depth in [m].
            focallength = prediction["focallength_px"]  # Focal length in pixels.

            # # plot
            # crop1 = depth.cpu().numpy()
            # plt.imshow(crop1)
            # plt.axis('off')  # Hide axes for better visualization
            # plt.savefig(
            #     f'/cache/depth_output/test_image_pairs_batch_{1}.png')
            # plt.show()
            # # until here

            depths = []
            for i, p in enumerate(results):
                # dpth = depth[p[1] : p[3], p[0] : p[2]].cpu().numpy()
                # #depths.append((np.mean(dpth), np.var(dpth)))
                # bbox_depth = depth * (embs0[i] > 0)
                # #Z3d = calculate_mean_3d_coordinates(p, bbox_depth.cpu(), focallength.cpu())
                a = compute_depth_histogram(p, depth.cpu().numpy())
                depths.append(a)
                # # plot
                # crop1 = bbox_depth.cpu().numpy()
                # plt.imshow(crop1)
                # plt.axis('off')  # Hide axes for better visualization
                # plt.savefig(
                #     f'/cache/depth_output/test_image_pairs_batch_{2}.png')
                # plt.show()
                # # until here

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        embs = embs.cpu().numpy()

        depths = np.array(depths)
        # Normalize the array along the last axis
        normalized_depths = depths / np.linalg.norm(depths, ord=2, axis=-1, keepdims=True)

        self.cache[tag] = embs#, mask_embs, normalized_depths
        return embs, (mask_embs[0].numpy(), mask_embs[1].numpy()), normalized_depths

    def initialize_model(self):

        sam2_checkpoint = os.getenv("SAM2_CHECKPOINT")
        model_cfg = os.getenv("MODEL_CFG")

        self.model_mask = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')

        # load Depth model:
        self.model_depth, self.model_depth_transf = depth_pro.create_model_and_transforms()
        self.model_depth.eval()

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
        elif self.dataset == "sport":
            path = "external/weights/sports_sbs_S50.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def _get_general_model(self):

        sam2_checkpoint = os.getenv("SAM2_CHECKPOINT")
        model_cfg = os.getenv("MODEL_CFG")

        self.model_mask = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
        # load Depth model:
        self.model_depth, self.model_depth_transf = depth_pro.create_model_and_transforms()
        self.model_depth.eval()

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
        # pass
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)


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

def compute_depth_histogram(bbox, depth_values, bins=1000):
    # hist, _ = np.histogram(depth_values, bins=bins, range=(depth_values.min(), depth_values.max()), density=True)

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Crop the depth map using the bounding box
    cropped_depth = depth_values[y_min:y_max, x_min:x_max]

    # Flatten the cropped depth map to get all depth values in the region
    cropped_depth_values = cropped_depth.flatten()
    org_depth_values = depth_values.flatten()

    # Calculate the histogram for the cropped depth values
    hist, _ = np.histogram(cropped_depth_values, bins=bins,
                           range=(cropped_depth_values.min(), cropped_depth_values.max()), density=True)

    return hist

def compute_depth_KDE(bbox, depth_values, bw_method='scott'):
    """
    Compute the Kernel Density Estimation (KDE) of depth values within a given bounding box.

    Parameters:
    bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).
    depth_values (np.ndarray): Depth values with shape (H, W).
    bw_method (str or float): Method for bandwidth calculation (default is 'scott').

    Returns:
    np.ndarray: Evaluated KDE values at specific depth points.
    """
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Crop the depth map using the bounding box
    cropped_depth = depth_values[y_min:y_max, x_min:x_max]

    # Flatten the cropped depth map to get all depth values in the region
    cropped_depth_values = cropped_depth.flatten()

    # Remove NaN values to avoid issues in KDE calculation
    cropped_depth_values = cropped_depth_values[~np.isnan(cropped_depth_values)]

    # Calculate KDE for the cropped depth values
    kde = gaussian_kde(cropped_depth_values, bw_method=bw_method)

    # # Use the minimum and maximum values of the total depth map for depth range
    # total_min_depth = np.nanmin(depth_values)
    # total_max_depth = np.nanmax(depth_values)
    depth_range = np.linspace(cropped_depth_values.min(), cropped_depth_values.max(), 100)

    # Evaluate the KDE on the depth range
    kde_values = kde(depth_range)

    return kde_values  # Only return the KDE values