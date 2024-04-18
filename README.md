# Speech Emotion Analyser

  # What is Speech Emotion Recognition?
  Human speech contains several features that the listener interprets to unpack the rich information transmitted by the speaker. The speaker also inadvertently shares tone, energy, speed, and other acoustic properties, which helps capture the subtext or intention and literal words.

# How does Speech Emotion Recognition Work?
Scientists apply various audio processing techniques to capture this hidden layer of information that can amplify and extract tonal and acoustic features from speech. Converting audio signals into numeric or vector format is not as straightforward as images. The transformation 

 # Open Source Speech Emotion Recognition Datasets

   Toronto emotional speech set (TESS) Collection
   https://tspace.library.utoronto.ca/handle/1807/24487

   The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
   https://zenodo.org/record/1188976

   # Citations
   Article

   https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
   
   https://medium.com/@diego-rios/speech-emotion-recognition-with-convolutional-neural-network-ae5406a1c0f7



   ## Overview:
This project aims to classify emotions from speech using Convolutional Neural Networks (CNNs). Emotion recognition is a vital component in various applications such as human-computer interaction, sentiment analysis, and customer service. By analyzing speech signals, this model predicts the emotional state of the speaker.

## Dataset:
- RAVDESS :- This dataset contains audio files of 24 actors speaking different sentences with various emotions.
- TESS :-This datasetcontains audio files of emotional speech from two actors.
- 
## Dependencies:

- Python 3
- Anaconda (recommended for managing Python environments)
- Jupyter Notebook (for running the provided notebook)
- TensorFlow (for building and training the CNN model)
- Keras (high-level neural networks API, running on top of TensorFlow)
- Scikit-learn (for evaluating model performance)
- Matplotlib and Seaborn (for data visualization)
- Pandas (for data manipulation)


## Model Architecture:
The CNN model architecture consists of multiple convolutional layers followed by max-pooling layers. Dropout layers are added to prevent overfitting. The model concludes with fully connected layers with ReLU activation and a softmax layer for multi-class classification.

## Training:
The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training is performed for 25 epochs with a batch size of 64. Training and validation accuracy metrics are monitored to evaluate the model's performance.

## Evaluation:
The trained model's performance is evaluated using both training and testing datasets. Metrics such as accuracy are calculated and visualized to assess the model's effectiveness.

## Confusion Matrix:
A confusion matrix is generated to visualize the model's performance across different emotions. It provides insights into the model's ability to correctly classify each emotion and identifies any misclassifications.

## Saving the Model:
The trained CNN model is saved in two parts: the model architecture is saved as a JSON file (capstone_project_emotion_detection_final_version.json), and the model weights are saved as an HDF5 file (capstone_project_emotion_detection_final_version.weights.h5). These files allow for easy deployment and reuse of the trained model.

## Usage:

1. Ensure all dependencies are installed (requirements.txt can be provided for easy setup).
2. Run the provided Jupyter Notebook (speech_emotion_final_notebook.ipynb) to train the CNN model and evaluate its performance.
3. Modify hyperparameters or experiment with different model architectures for further optimization.
4. Deploy the saved model in your application for real-time emotion detection from speech signals.


## Contributors:

- Dave Burgman
- April Holmes
- Neil Lawren
- Ipsita Pattanaik
  

  
   


