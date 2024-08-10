# Skin Disease Classification Project

This project aims to classify various skin diseases using machine learning techniques. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset).

## Dataset

The dataset can be found [here](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset). It includes images and metadata of various skin diseases. The dataset is structured into several categories, each representing a different type of skin disease.

### Dataset Structure

The data is organized in the following folder structure:
```
~/Git/MachineLEarning-School/SkinDiseases/skin-disease-dataset/
│
├── BA-cellulitis
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── BA-impetigo
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── FU-athlete-foot
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── FU-nail-fungus
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── FU-ringworm
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── PA-cutaneous-larva-migrans
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
├── VI-chickenpox
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
└── VI-shingles
    ├── image1.jpeg
    ├── image2.jpeg
    └── ...
```

Each folder contains `.jpeg` images representing a specific type of skin disease.

## Project Structure

- `data/`: Contains the dataset images and metadata.
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, and model training.
- `models/`: Saved models and checkpoints.
- `src/`: Source code for data processing, model training, and evaluation.
- `reports/`: Generated analysis and reports.

## Setup

### Requirements

- Python 3.8+
- Jupyter Notebook
- TensorFlow / PyTorch
- Scikit-learn
- Pandas
- Matplotlib
- NumPy

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/robertnesterodhiambo/MachineLEarning-School.git
    cd MachineLEarning-School/SkinDiseases
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure the dataset is in the `~/Git/MachineLEarning-School/SkinDiseases/skin-disease-dataset/` directory as described above.

## Usage

### Data Exploration and Preprocessing

Open and run the Jupyter notebooks in the `notebooks/` directory to explore and preprocess the data.

### Training

To train the model, run:
```bash
python src/train.py
```

### Evaluation

To evaluate the model, run:
```bash
python src/evaluate.py
```

### Inference

For inference on new images, run:
```bash
python src/inference.py --image_path path_to_image
```

## Results

The results of the model training and evaluation, including performance metrics and visualizations, are saved in the `reports/` directory.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- Thanks to the authors of the [Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) for providing the data.