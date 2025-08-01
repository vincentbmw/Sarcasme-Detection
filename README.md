# Sarcasm Detection Model

A deep learning project for detecting sarcasm in text using [TensorFlow](https://www.tensorflow.org/) and [Universal Sentence Encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual-large). This model can be used for sentiment analysis applications to better understand the true intent behind textual communications.

## Project Overview

Sarcasm detection is a challenging natural language processing task that involves identifying when someone is saying the opposite of what they mean. This project implements a neural network model that can:

- **Detect sarcastic vs non-sarcastic text** with high accuracy
- **Analyze sentiment** in social media posts, reviews, and comments
- **Help businesses** understand customer feedback more accurately
- **Improve chatbots** and automated response systems

## Use Cases

- **Product Review Analysis**: Identify when customers are being sarcastic in reviews
- **Social Media Monitoring**: Better understand public sentiment about brands/products
- **Customer Service**: Help support teams identify frustrated customers using sarcasm
- **Content Moderation**: Detect potentially negative content masked as sarcasm
- **Market Research**: More accurate sentiment analysis of consumer feedback

## Project Structure

```
├── dataset/
│   ├── raw_dataset/          # Original training, validation, and test data
│   ├── processed_dataset/    # Cleaned and balanced datasets
│   └── additional/           # Supporting files (KBBI dictionary, slang words)
├── embeddings/               # Pre-computed sentence embeddings
├── mymodels/                 # Trained model files
├── preprocessing.ipynb       # Data preprocessing and cleaning
├── sarcas_model_from_gemini.ipynb  # Main model training notebook
├── distribution.py           # Dataset balancing and cleaning script
├── check_distribution.py     # Dataset analysis script
└── requirements.txt          # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd sarcasm-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

This project combines three Indonesian sarcasm detection datasets from Hugging Face:

### Dataset Sources

- [w11wo/twitter_indonesia_sarcastic](https://huggingface.co/datasets/w11wo/twitter_indonesia_sarcastic) - Indonesian Twitter posts
- [w11wo/reddit_indonesia_sarcastic](https://huggingface.co/datasets/w11wo/reddit_indonesia_sarcastic/tree/main/data) - Indonesian Reddit comments  
- [enoubi/Reddit-Indonesian-Sarcastic-Fix-Text](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Fix-Text) - Cleaned Reddit data

The combined dataset provides diverse Indonesian text from social media and forums, split into 80% training, 10% validation, and 10% test data.

### Data Preprocessing

The dataset undergoes several preprocessing steps:
- **Text cleaning**: Removing special characters, URLs, mentions
- **Class balancing**: Addressing imbalanced sarcastic vs non-sarcastic samples
- **Duplicate removal**: Ensuring data quality
- **Indonesian language support**: Using Sastrawi for Indonesian text processing

## Model Architecture

The model uses:
- **Universal Sentence Encoder**: For high-quality text embeddings
- **Dense Neural Network**: Multiple layers with dropout for classification
- **Class weighting**: To handle imbalanced datasets
- **Early stopping**: To prevent overfitting

### Key Features:
- Multilingual support (English and Indonesian)
- Efficient sentence embeddings
- Regularization techniques to improve generalization
- Comprehensive evaluation metrics



## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Universal Sentence Encoder by Google
- TensorFlow team for the excellent framework
- Various sarcasm detection datasets used for training


## 📝 License

This project is open source and available under the [MIT License](https://github.com/vincentbmw/Sarcasme-Detection/blob/main/LICENSE).
