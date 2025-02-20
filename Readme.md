# **Drug-Protein Interaction Prediction Using CLIP and Deep Learning**

## **Project Overview**
This project aims to predict drug-protein interaction strength by leveraging deep learning models, specifically OpenAI's CLIP model. The interaction strength is quantified by the inhibitory constant (*Ki* value), which measures how effectively a drug binds to a target protein. The model takes molecular structure images of drugs (converted from SMILES representations) and protein structure images as input to predict the *Ki* value.

---

## **Key Features**
- Converts **SMILES** strings into **2D molecular structure images**
- Generates **protein sequence logos** from amino acid sequences
- Implements a **CLIP-based deep learning model** for *Ki* value prediction
- Achieves **RMSE of 0.6041** on test set predictions
- Provides **comprehensive visualization** of results

---

## **Technical Stack**
- **PyTorch** for deep learning implementation
- **RDKit** for molecular visualization
- **Logomaker** for protein sequence visualization
- **CLIP (Contrastive Language-Image Pre-training) model**
- **Transformers** library from Hugging Face
- **scikit-learn** for data splitting and evaluation metrics
- **Pandas & Matplotlib** for data handling and visualization

---

## **Model Architecture**
- Utilizes **CLIP's vision encoder** for processing both molecular and protein images
- Extracted features are **combined and passed through a fully connected layer**
- Trained using **MSE loss function and AdamW optimizer**
- Implements **batch processing with GPU acceleration**

---

## **Key Components**

### **1. Dataset Preparation**
- The dataset (`output.csv`) contains **drug-protein pairs** with their corresponding *Ki* values.
- Each drug is represented as a **2D image derived from its SMILES notation**.
- Each protein is represented as a **structural image** generated from its sequence.
- Data is split into **training (70%), validation (15%), and test (15%)** sets.

### **2. Data Preprocessing**
- Image transformations ensure compatibility with **CLIP's vision encoder**.
- The dataset is handled using a PyTorch `Dataset` class for efficient loading.

### **3. Model Architecture**
- Built upon **OpenAI's CLIP (Vision Transformer-based model)**.
- Separate **CLIP vision encoders** process **drug and protein images**.
- Extracted features are **combined and passed through a fully connected layer** to predict *Ki* values.

### **4. Training Process**
- Trained using **Mean Squared Error (MSE) loss function**.
- **AdamW optimizer** with a learning rate of `1e-5`.
- Training occurs over **multiple epochs**, with validation loss monitoring.

### **5. Evaluation & Visualization**
- Evaluated on the **test set**, with a **scatter plot comparing predicted vs actual *Ki* values**.
- A **sorted line graph** helps visualize prediction trends.

### **6. Model Deployment & Saving**
- The trained model is saved as **`drug_protein_interaction_model.pth`** for future use.


## **Technologies & Libraries Used**
- **Deep Learning:** PyTorch, Transformers (CLIP Model)
- **Data Processing:** Pandas, NumPy, Matplotlib
- **Image Handling:** Torchvision, PIL
- **Training Acceleration:** GPU utilization with CUDA


## **Results**
- **Test Loss:** 0.3651
- **Mean Squared Error (MSE):** 0.3649
- **Root Mean Squared Error (RMSE):** 0.6041

---

## **Usage Guide**
1. **Prepare input data**: Convert SMILES strings and protein sequences.
2. **Generate molecular images**: Use RDKit to visualize molecular structures.
3. **Create protein sequence logos**: Utilize Logomaker for sequence representation.
4. **Train the model**: Train from scratch or use pre-trained weights.
5. **Make predictions**: Predict *Ki* values for new drug-protein pairs.

---

## **Visualizations**
The project includes various visualization tools:
- **Training & validation loss curves**
- **Scatter plots** comparing predicted vs actual *Ki* values
- **Line graphs** displaying sorted predicted and actual values





