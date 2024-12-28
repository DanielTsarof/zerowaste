from pycocotools import mask as maskUtils
from typing import Tuple


def decode_rle(rle_obj):
    """
    Декодирует RLE с помощью pycocotools.
    На вход ожидается словарь {'size': [height, width], 'counts': <string RLE>}.
    Возвращает 2D numpy-массив bool с формой (height, width).
    """
    return maskUtils.decode(rle_obj)


def get_area_from_segmentation(segm):
    """
    Возвращает число пикселей (площадь) в сегментации.
    segm может быть:
      1) RLE в dict {'size': [h, w], 'counts': "..."},
      2) Полигон (или список полигонов) [[x1, y1, x2, y2, ...], ...].
    """
    if isinstance(segm, dict) and "counts" in segm and "size" in segm:
        # RLE
        rle = segm
    else:
        # pycocotools.mask.frPyObjects --> RLE
        # segm может быть списком полигонов или одним полигоном
        rle = maskUtils.frPyObjects(segm, 1080, 1920)  # Все изображения в датасете однгого размера

        # Если segm - это список полигонов, frPyObjects вернёт список RLE,
        # тогда нужно объединить (merge)
        if isinstance(rle, list):
            rle = maskUtils.merge(rle)

    # декодируем RLE
    mask = maskUtils.decode(rle)  # numpy array shape (h, w), dtype=uint8
    area = mask.sum()
    return area


def area_from_annotation(ann, img_hw: Tuple[int, int]):
    segm = ann["segmentation"]
    h, w = img_hw[ann["image_id"]]
    if isinstance(segm, dict) and "counts" in segm:
        # Это RLE уже включает "size": [h, w]
        return maskUtils.area(segm)
    else:
        # Это полигон (или список полигонов).
        # Создаём RLE из polygons:
        rle = maskUtils.frPyObjects(segm, h, w)
        if isinstance(rle, list):
            rle = maskUtils.merge(rle)
        return maskUtils.area(rle)  # float
