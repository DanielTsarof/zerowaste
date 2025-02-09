from pycocotools import mask as maskUtils
from typing import Tuple
import matplotlib.pyplot as plt


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


def visualize_annotation(image_path, label_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(label_path) as f:
        lines = f.readlines()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = np.array([list(map(float, p.split(','))) for p in parts[1:]])
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
        plt.scatter(points[:, 0], points[:, 1], s=40, c='blue')
        plt.title(f'Class {class_id}', fontsize=14)
    
    plt.axis('off')
    plt.show()