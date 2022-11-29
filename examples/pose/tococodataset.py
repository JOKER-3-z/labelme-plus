#pip install https://github.com/cm107/annotation_utils/archive/master.zip
# https://github.com/cm107/annotation_utils


from common_utils.file_utils import file_exists
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler, COCO_Category
from annotation_utils.labelme.structs import LabelmeAnnotationHandler

# Define Labelme Directory Paths
img_dir = '/home/xiaoran/work/pose/dataset/zijian/test'
json_dir = '/home/xiaoran/work/pose/dataset/zijian/test'

# Load Labelme Handler
labelme_handler = LabelmeAnnotationHandler.load_from_dir(load_dir=json_dir)

# Define COCO Categories Before Conversion
if not file_exists('categories_example.json'): # Save a new categories json if it doesn't already exist.
    categories = COCO_Category_Handler()
    categories.append( # Standard Keypoint Example
        COCO_Category(
            id=len(categories),
            supercategory='person',
            name='person',
            keypoints=[ # The keypoint labels are defined here
                "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                "right_knee", "left_ankle", "right_ankle"
            ],
            skeleton=[ # The connections between keypoints are defined with indecies here
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                 [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        )
    )
    categories.append( # Simple Keypoint Example
        COCO_Category.from_label_skeleton(
            id=len(categories),
            supercategory='pet',
            name='cat',
            label_skeleton=[
                ['left_eye', 'right_eye'],
                ['mouth_left', 'mouth_center'], ['mouth_center', 'mouth_right']
            ]
        )
    )
    for name in ['duck', 'sparrow', 'pigion']:
        categories.append( # Simple Non-Keypoint Example
            COCO_Category(
                id=len(categories),
                supercategory='bird',
                name=name
            )
        )
    categories.save_to_path('categories_example.json')
else: # Or load from an existing categories json
    categories = COCO_Category_Handler.load_from_path('categories_example.json')

# Convert To COCO
coco_dataset = COCO_Dataset.from_labelme(
    labelme_handler=labelme_handler,
    categories=categories,
    img_dir=img_dir
)
coco_dataset.save_to_path(save_path='converted_coco.json', overwrite=True)
coco_dataset.display_preview(show_details=True) # Optional: Preview your resulting dataset.