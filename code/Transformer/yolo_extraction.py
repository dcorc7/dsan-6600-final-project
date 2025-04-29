import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Compute IoU between two arrays of N×4 and M×4 boxes
def calculate_iou(boxes1, boxes2):
    # boxes1: (N,4), boxes2: (M,4)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

def process_videos(test_mode=False):
    RELEVANT_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    class_map = {cls_id: idx for idx, cls_id in enumerate(RELEVANT_CLASSES)}
    C = len(RELEVANT_CLASSES)

    # load labels once
    labels_df = pd.read_csv('data/train_labels.csv', dtype={'id': str})
    labels_df['id'] = labels_df['id'].str.zfill(5)

    model = YOLO('yolov8n.pt')               

    os.makedirs('annotated_frames', exist_ok=True)

    root_dir  = 'cleaned_frames/train'
    video_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    if test_mode:
        video_dirs = video_dirs[:3]
        print("Test mode: only processing first 3 videos")

    for vid in tqdm(video_dirs, desc="Videos"):
        vid_in  = os.path.join(root_dir, vid)
        vid_out = os.path.join('annotated_frames', vid)
        os.makedirs(vid_out, exist_ok=True)

        # read video label
        try:
            label = int(labels_df.loc[labels_df['id'] == vid, 'target'].iloc[0])
        except IndexError:
            raise ValueError(f"No label found for video ID {vid}")

        frames = sorted(f for f in os.listdir(vid_in) if f.endswith('.jpg'))
        frame_feats = []
        prev_boxes  = None

        for i, frame_name in enumerate(frames, start=1):
            path = os.path.join(vid_in, frame_name)
            res  = model(path, verbose=False)[0]

            # always get numpy arrays—even if empty
            xxyy  = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)

            # filter classes
            mask = np.isin(clses, RELEVANT_CLASSES)
            xxyy, confs, clses = xxyy[mask], confs[mask], clses[mask]

            # coords: guard empty
            if xxyy.size:
                x1, y1 = xxyy[:, 0], xxyy[:, 1]
                x2, y2 = xxyy[:, 2], xxyy[:, 3]
            else:
                x1 = y1 = x2 = y2 = np.array([])

            widths  = x2 - x1
            heights = y2 - y1
            areas   = widths * heights
            aratios = widths / (heights + 1e-6)
            x_cent  = (x1 + x2) / 2
            y_cent  = (y1 + y2) / 2

            # build feature vector
            feat = []

            # temporal deltas
            counts = np.zeros(C)
            for c in clses:
                counts[class_map[c]] += 1

            if frame_feats:
                prev = frame_feats[-1]
                feat.extend(counts - prev[:C])
                prev_pos = prev[-4:]
                feat.extend([
                    x_cent.mean() - prev_pos[0],
                    y_cent.mean() - prev_pos[1],
                    x_cent.std()  - prev_pos[2],
                    y_cent.std()  - prev_pos[3]
                ])
            else:
                feat.extend([0.0] * (C + 4))

            # counts
            feat.extend(counts.tolist())

            # global stats
            feat.extend([
                confs.mean() if confs.size else 0.0,
                areas.mean() if areas.size else 0.0,
                aratios.mean() if aratios.size else 0.0,
                x_cent.mean() if x_cent.size else 0.0,
                y_cent.mean() if y_cent.size else 0.0,
                x_cent.std()  if x_cent.size else 0.0,
                y_cent.std()  if y_cent.size else 0.0,
            ])

            # motion IoU
            if prev_boxes is not None and xxyy.size and prev_boxes.size:
                iou_m = calculate_iou(xxyy, prev_boxes)
                feat.extend([iou_m.mean(), iou_m.max()])
            else:
                feat.extend([0.0, 0.0])
            prev_boxes = xxyy.copy()

            # density
            if x_cent.size > 1:
                dists = np.sqrt((x_cent[:, None] - x_cent)**2 + (y_cent[:, None] - y_cent)**2)
                iu = np.triu_indices_from(dists, k=1)
                pdist = dists[iu]
                feat.extend([pdist.mean(), pdist.min(), pdist.size / (len(x_cent)*(len(x_cent)-1)/2)])
            else:
                feat.extend([0.0, 0.0, 0.0])

            # size percentiles
            if areas.size:
                feat.extend([
                    np.percentile(areas, 25),
                    np.percentile(areas, 50),
                    np.percentile(areas, 75),
                    areas.max() / (areas.min() + 1e-6)
                ])
            else:
                feat.extend([0.0] * 4)

            frame_feats.append(np.array(feat, dtype=float))

        # stack into array
        raw_arr = np.stack(frame_feats)  # (F, D)
        norm_arr = raw_arr.copy()

        # per‑feature normalization
        for j in range(norm_arr.shape[1]):
            col = norm_arr[:, j]
            if np.any(col != 0):
                mu, sd = col.mean(), col.std() + 1e-6
                norm_arr[:, j] = (col - mu) / sd

        # temporal smoothing
        k = 3
        if norm_arr.shape[0] > k:
            kernel = np.ones(k) / k
            for j in range(norm_arr.shape[1]):
                norm_arr[:, j] = np.convolve(norm_arr[:, j], kernel, mode='same')

        # save
        np.save(os.path.join(vid_out, 'frame_data_raw.npy'), raw_arr)
        np.save(os.path.join(vid_out, 'frame_data.npy'), norm_arr)
        np.save(os.path.join(vid_out, 'label.npy'), [label])

def main():
    process_videos(test_mode=False)

if __name__ == '__main__':
    main()

# quick check
print(np.load('annotated_frames/00000/frame_data.npy'))
print(np.load('annotated_frames/00000/frame_data.npy').shape)
