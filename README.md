**Landscape Image Classifier**

**This project is a deep learning-based image classifier capable of predicting five landscape classes:**  
- **Mountains**
- **Coasts**
- **Deserts**
- **Glaciers**  
- **Forests**     
   
---   
  
**Features**   
* **Multi-class Classification**: Classifies images into one of five landscape categories.<br/>
* **Data Augmentation**: Enhances model robustness with augmented training data.<br/>
* **Model Evaluation**: Provides performance metrics including accuracy, loss, and confusion matrix visualization.<br/>
* **K-fold Cross-Validation**: Ensures reliability and generalizability of results.<br/>
* **Efficient Preprocessing**: Optimized dataset pipeline for resizing, rescaling, and augmentation.<br/>

<br/> 
  
**Project Files**   
 
* **index.py** :  Streamlit Application for Classification of Landscape Images.<br/> 
* **resnet91.keras** :  Trained model weights.<br/>
* **requirements.txt** :  Contains all the dependencies required for the application<br/>


<br/>
 
**Requirements**<br/> 
Ensure you have the following dependencies installed on your system:<br/>

* Python 3.8 or higher<br/>
* Libraries required for this project are mentioned in the requirements.txt<br/>

**Installation**<br/>
* **Clone this repository**:<br/>
[https://github.com/NAKULMAK05/Landscapes-Image-Classifier.git]<br/>
<br/>

* **Install the required Python libraries**:<br/>
pip install -r requirements.txt<br/>

<br/>

**How to Run the App**<br/>
run the streamlit application index.py and then enter command streamlit run index.py in the terminal .<br/>
the streamlit application will be deployed in the browser where you can classify various landscape images Categories and Use the predicions for your further research.<br/>
  <br/>
<br/>
 
**Future Enhancements**<br/>
* Expand the dataset to include more landscape types.<br/>
* Improve the model's accuracy with hyperparameter tuning.<br/>
* Develop a web-based or mobile app interface for the classifier.<br/>
* Integrate Grad-CAM to visualize the model's decision-making process.<br/>

<br/>

**Technologies Used**<br/>

* **Python**: Programming language<br/>
* **TensorFlow/Keras**: Deep learning framework<br/>
* **Matplotlib/Seaborn**: Visualization tools<br/>
* **NumPy/Pandas**: Data manipulation<br/>

<br/> 

**Acknowledgments**<br/>
* **Dataset**: The dataset used in this project was sourced from Kaggle (Landscape Images dataset of 12k images).<br/>
* **Libraries**: Special thanks to the open-source libraries that made this project possible.<br/>

<br/>

**License**<br/>
This project is licensed under the [MIT License](LICENSE).

