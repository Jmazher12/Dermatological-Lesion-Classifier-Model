# Dermatological Lesion Classification Model

This project focuses on building a machine learning model to classify dermatological lesions into benign and malignant categories using image data. The primary goal is to assist in early detection and diagnosis of skin conditions, potentially aiding healthcare professionals in making more informed decisions.

---

## Project Overview

Skin cancer is one of the most common types of cancer worldwide, and early detection is critical for effective treatment. This project leverages deep learning techniques to analyze dermatological images and classify lesions based on their characteristics.

The dataset used contains images of skin lesions along with corresponding labels indicating whether the lesion is benign or malignant. The project includes data preprocessing, augmentation, model training, evaluation, and visualization of results.

---

## Key Features

- **Custom Data Generator**: A custom data generator was implemented to preprocess images, overlay lesion masks, and normalize input data for the model.
- **Deep Learning Model**: A convolutional neural network (CNN) architecture was trained to classify lesions based on image features.
- **Data Augmentation**: Techniques such as flipping, rotation, scaling, and normalization were applied to enhance model generalization.
- **Evaluation Metrics**: The model's performance was assessed using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
- **Challenges Addressed**:
  - High inter-class similarity between benign and malignant lesions.
  - Limited dataset size necessitating augmentation and careful preprocessing.

---

## Project Workflow

1. **Data Preparation**:
   - Loaded a dataset containing dermatological images and their corresponding labels.
   - Preprocessed the images by resizing them to a consistent target size and normalizing pixel values.
   - Used lesion masks to overlay additional information onto the input images.

2. **Model Training**:
   - Designed a CNN-based architecture for image classification.
   - Trained the model using the preprocessed dataset with augmented data.

3. **Evaluation**:
   - Evaluated the trained model on a validation set.
   - Generated a confusion matrix to analyze classification performance.

4. **Results**:
   - The initial model struggled with classifying malignant lesions due to limited data and high similarity between classes.
   - Recommendations for improvement include increasing dataset size, leveraging transfer learning, and utilizing GPU resources for faster training.

---

## Challenges

- **Data Scarcity**: The dataset was relatively small, which limited the model's ability to generalize effectively. This led to overfitting or underfitting during training.
- **Class Imbalance**: Benign lesions were more prevalent in the dataset compared to malignant ones, leading to biased predictions.
- **Computational Resources**: Training deep learning models on large image datasets requires significant computational power. Limited resources slowed down experimentation.

---

## Future Improvements

1. Expand the dataset by incorporating publicly available dermatological image datasets or collaborating with medical institutions.
2. Use transfer learning with pre-trained models like ResNet or EfficientNet for better feature extraction.
3. Implement advanced techniques like attention mechanisms or multimodal learning (e.g., combining clinical metadata with image data).
4. Train the model on GPU-enabled hardware for faster convergence and experimentation.

---

## Usage

This repository contains the following files:
- `lesion_classification.ipynb`: The main Jupyter Notebook with code for preprocessing, training, and evaluation.
- `data/`: Directory containing sample images used in this project (not included due to size constraints; refer to dataset sources below).
- `README.md`: This file describing the project.

### Steps to Run:

1. Clone the repository: git clone https://github.com/Jmazher12/dermatological-lesion-classification.git
2. Install all dependencies
3. Open `Dermatological_Lesion_Classification_Model.ipynb` in Jupyter Notebook or JupyterLab and run all code.

---

## Dataset

The dataset used in this project is:

- https://www.fc.up.pt/addi/ph2%20database.html
- This weblink will take you to the site where you can download the 'PH2 Dataset'
- Alternatively, you can go to the 'Code' section to find a link to a Google Drive folder containing the data.

---

## Acknowledgments

This project was inspired by ongoing efforts in medical AI research aimed at improving diagnostic tools for healthcare professionals. Special thanks to open-source contributors who provide datasets and tools for medical image analysis.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

