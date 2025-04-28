# Title  
Long Leg Radiograph Analysis

# Team members  
Matthew Boyd (mattboyd23), Sunishka Deshpande (sunishkad)

# Project description  
We are developing a convolutional neural network (CNN) to automate the detection of key anatomical landmarks in long leg radiographs (LLRs), a common imaging modality used in orthopedic assessments. Our model will be trained to accept full-length LLR images and output the (x, y) pixel coordinates of three clinically relevant landmarks: the center of the femoral head, the center of the knee joint (patella), and the center of the ankle joint
These landmarks are foundational in calculating critical measurements such as leg length, hip-knee-ankle (HKA) angle, and mechanical axis deviation, which are essential in diagnosing and planning treatment for musculoskeletal deformities and alignment issues.
- **Dataset**: We will be using the publicly available Osteoarthritis Initiative (OAI) dataset, which provides a large volume of labeled radiographic images suitable for training and evaluating our model. We will use ITK-SNAP, an open-source medical imaging annotation tool, to manually label the anatomical landmarks in each image to generate ground truth data for training the model.
- **Goal**: The convolutional neural network should be trained to accept long leg radiograph images and output coordinate locations of several anatomical landmarks, including the center of the femoral head, center of the knee joint (patella), and center of the ankle joint.

# Model Proposals
Below is a high-level sumary of the two candidate models with their corresponding folder names. Proposal 1 is our selected model, while Proposal 2 showcases a separate model developed that did not have better performance.

**Proposal 1: Original CNN**
- **Folder:** proposal1-original_CNN
- **Architecture:** Four 5x5 convolutional layers with max-pooling, two fully-connected layers, coordinate regression head
- **Output:** Sigmoid-activated normalized (x, y) in [0, 1]
- **Loss:** Pixel-space SmoothL1 (Huber) on denormalized coordinates
- **Training:** Tiny translate/scale augmentations, AdamW optimizer, cosine LR decay, per-landmark RMSE logging
-**Use case:** Baseline end-to-end CNN for direct comparison

**Proposal 2: ResNet-18 Backbone**
- **Folder:** proposal2-resnet18_CNN
- **Architecture:** Pretrained ResNet-18 (grayscale adaptor), freeze low-level features, lightweight regression head
- **Output:** Sigmoid-activated normalized (x, y) in [0, 1]
- **Loss:** Pixel-space SmoothL1 (Huber) on denormalized coordinates
- **Training:** Clinically-safe augmentations (translate/scale + brightness/contrast), AdamW, CosineAnnealingWarmRestarts, gradient clipping, detailed per-landmark RMSE logging
-**Use case:** Enhanced feature extractor to boost localization accuracy and robustness
