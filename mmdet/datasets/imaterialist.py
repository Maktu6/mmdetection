import numpy as np
from pycocotools.coco import COCO

# from .custom import CustomDataset
from .coco import CocoDataset


class iMaterialistDataset(CocoDataset):

    CLASSES = ('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 
             'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 
             'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 
             'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 
             'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 
             'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 
             'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel')

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if with_mask:
            gt_masks = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
        if with_mask:
            # RLEs to mask, build a generator for saving memory
            gt_masks = map(self.coco.annToMask, ann_info) 
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
        return ann

    # def load_annotations(self, ann_file):
    #     self.coco = COCO(ann_file)
    #     self.cat_ids = self.coco.getCatIds()
    #     self.cat2label = {
    #         cat_id: i + 1
    #         for i, cat_id in enumerate(self.cat_ids)
    #     }
    #     self.img_ids = self.coco.getImgIds()
    #     img_infos = []
    #     for i in self.img_ids:
    #         info = self.coco.loadImgs([i])[0]
    #         info['filename'] = info['file_name']
    #         img_infos.append(info)
    #     return img_infos

    # def get_ann_info(self, idx):
    #     img_id = self.img_infos[idx]['id']
    #     ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    #     ann_info = self.coco.loadAnns(ann_ids)
    #     return self._parse_ann_info(ann_info, self.with_mask)

    # def _filter_imgs(self, min_size=32):
    #     """Filter images too small or without ground truths."""
    #     valid_inds = []
    #     ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
    #     for i, img_info in enumerate(self.img_infos):
    #         if self.img_ids[i] not in ids_with_ann:
    #             continue
    #         if min(img_info['width'], img_info['height']) >= min_size:
    #             valid_inds.append(i)
    #     return valid_inds
