# CAUTION: Dataset Usage

This document outlines important warnings and guidelines about using datasets with this project. Please ensure that your dataset adheres to the preprocessing requirements to avoid errors or unexpected behavior.

---

## 1. Preprocessed Data Requirement
- The model in this project has been trained and tested on **preprocessed datasets**, not raw data.
- **Raw data downloaded from the provided link must be preprocessed** to match the input requirements of the model.

---

## 2. Preprocessing Guidelines
To ensure the uploaded dataset works correctly:
1. **Image Dimensions**: All images must be resized to `150x150 pixels`. 
2. **Normalization**: Pixel values should be scaled to the range `[0, 1]` (e.g., dividing raw pixel values by 255).
3. **File Format**: Supported formats include `.jpg`, `.jpeg`, and `.png`. Other formats may cause errors.
4. **Class Labels**: Ensure that the data includes only the following classes:
   - Apple
   - Cabbage
   - Carrot
   - Cucumber
   - Eggplant
   - Pear
5. **Directory Structure**: If you're using a folder of images for classification, each category should be in its respective subdirectory.

---

## 3. Risks of Using Raw Data
Uploading raw data directly without preprocessing may lead to:
- **Incorrect Predictions**: Images not aligned with preprocessing standards may produce unreliable results.
- **Application Errors**: Mismatched image dimensions or unsupported formats can cause the application to crash or behave unexpectedly.
- **Reduced Accuracy**: Differences in color profiles, lighting conditions, or noise in raw data may degrade performance.

---