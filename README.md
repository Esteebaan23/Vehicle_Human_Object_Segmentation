# Vehicle_Human_Object_Segmentation

# Vehicle, Human, and Static Object Segmentation in Urban Areas

This project focuses on the segmentation of vehicles, humans, and static objects in urban areas of the city. The dataset used includes annotations of the following type:

- **Dataset Example with Annotation**:

  ![Dataset Annotation](path/to/dataset_annotation_image.png)

## Methodology

Each folder contains scripts that generate masks or annotations in the required format for various models. The following are examples of format requirements:

- **YOLO v11 Segmentation**: Requires a `.txt` file containing points.
- **U-Net and SAM**: Require masks generated for each class.

**Note**: Certain models require images to be resized. For instance, YOLO requires input images to be 640 pixels.

## Results

The results for each model are shown in the following figures:

- **YOLO Results**:

  ![YOLO Results](path/to/yolo_results_image.png)

- **U-Net Results**:

  ![U-Net Results](path/to/unet_results_image.png)

- **SAM Results**:

  ![SAM Results](path/to/sam_results_image.png)

Replace `path/to/...` with the actual paths to your images.

