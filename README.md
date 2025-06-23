# simulated-av-lane-assist
A simulated autonomous driving project implementing a lane-keeping assistant using a hybrid Python-C++ architecture with MLOps principles.

## Project Structure

```
├── .gitignore
├── README.md
├── app/
├── data/
├── docs/
├── models/
└── src/
    ├── cpp/
    └── python/
```

## Dataset

This project utilises a custom-generated synthetic driving dataset, specifically designed for training machine learning models for autonomous vehicle lane-keeping.

The dataset comprises **34,379 grayscale images** captured from a simulated front-facing camera, covering driving scenarios on both **straight road segments** and a **curved (left-turn) arc**. Each image is paired with a corresponding steering angle label.

**Access the complete dataset on Kaggle:**
[Link to the Kaggle Dataset Here] 
(`https://www.kaggle.com/datasets/afsanehm/simulated-driving-data`)

For a detailed explanation of the data generation process, including the specific parameters used for diversification and the dataset's internal structure, please refer to the dedicated [`Data_README.md`](data/README.md).

### ML Model Training & Accessibility

The deep learning model was trained and evaluated on a dedicated Kaggle notebook: [Deep Learning for Simulated Driving](https://www.kaggle.com/code/afsanehm/deep-learning-for-simulated-driving). The trained model checkpoints (including the best-performing version) are accessible within this repository in the `models/` directory.


