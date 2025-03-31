# Title  
Long Leg Radiograph Analysis

## Team members  
Matthew Boyd (mattboyd23), Sunishka Deshpande (sunishkad)

# Project description  
We are developing a convolutional neural network (CNN) to automate the detection of key anatomical landmarks in long leg radiographs (LLRs), a common imaging modality used in orthopedic assessments. Our model will be trained to accept full-length LLR images and output the (x, y) pixel coordinates of three clinically relevant landmarks: the center of the femoral head, the center of the knee joint (patella), and the center of the ankle joint
These landmarks are foundational in calculating critical measurements such as leg length, hip-knee-ankle (HKA) angle, and mechanical axis deviation, which are essential in diagnosing and planning treatment for musculoskeletal deformities and alignment issues.
- **Dataset**: We will be using the publicly available Osteoarthritis Initiative (OAI) dataset, which provides a large volume of labeled radiographic images suitable for training and evaluating our model.
- **Goal**: The convolutional neural network should be trained to accept long leg radiograph images and output coordinate locations of several anatomical landmarks, including the center of the femoral head, center of the knee joint (patella), and center of the ankle joint.
