# Vehicle_Human_Object_Segmentation

# Vehicle, Human, and Static Object Segmentation in Urban Areas

This project focuses on the segmentation of vehicles, humans, and static objects in urban areas of the city. The dataset used includes annotations of the following type:

- **Dataset Example with Annotation**:

  ![Dataset Annotation](Images/input.png)

## Methodology

Each folder contains scripts that generate masks or annotations in the required format for various models. The following are examples of format requirements:

- **YOLO v11 Segmentation**: Requires a `.txt` file containing points.
- **U-Net and SAM**: Require masks generated for each class.

**Note**: Certain models require images to be resized. For instance, YOLO requires input images to be 640 pixels.

## Results

The results for each model are shown in the following figures:

- **YOLO Results**:

  <div style="display: flex; justify-content: space-between;">
    <img src="Images/overlay_predicted_yolo1.png" alt="YOLO Result 1" style="width: 48%;">
    <img src="Images/overlay_predicted_yolo5.png" alt="YOLO Result 2" style="width: 48%;">
  </div>

- **U-Net Results**:

  <div style="display: flex; justify-content: space-between;">
    <img src="Images/overlay_predicted_unet1.png" alt="YOLO Result 1" style="width: 48%;">
    <img src="Images/overlay_predicted_unet5.png" alt="YOLO Result 2" style="width: 48%;">
  </div>

- **SAM Results**:

  <div style="display: flex; justify-content: space-between;">
    <img src="Images/overlay_predicted_sam1.png" alt="YOLO Result 1" style="width: 48%;">
    <img src="Images/overlay_predicted_sam5.png" alt="YOLO Result 2" style="width: 48%;">
  </div>





