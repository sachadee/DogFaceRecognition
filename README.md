## The most accurate dog face recognitin!!!

# ~96.7% on the DogFaceNet dataset!!
# Live Demo : [Hugging space](https://huggingface.co/spaces/SachaDee/DogFaceRecognition)
This repository contains the implementation of a DogFace recognition system using ONNX and TFLite models trained on the [DogFaceNet](https://zenodo.org/records/12578449) dataset. The training process is based on a **Siamese network** using **triplet loss**, which enables the model to distinguish between different dog faces effectively.
a siamese network with Triplet loss

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Model Architecture](#model-architecture)
-   [Triplet Loss](#triplet-loss)
-   [Ideal Use Case](#ideal-use-case)
-   [References](#references)

## Project Overview

The goal of this project is to create a model capable of recognizing individual dogs based on their facial features. The model is designed to support two formats:

-   [**ONNX**](https://onnx.ai/) for cross-platform machine learning.
-   [**TFLite**](https://www.tensorflow.org/lite/guide) for lightweight deployment on mobile and edge devices.

The system is trained using a **Siamese triplet network** and can:

1.  Identify if two dog face images belong to the same dog (1:1 matching).
2.  Find a lost dog by comparing it to a database of street dog images (1
    
    matching).

## Dataset

We use the [DogFaceNet](https://zenodo.org/records/12578449) dataset, which consists of dog face images for various breeds. The dataset is divided into:

-   Training set: Images of dogs labeled by their breed.
-   Testing set: Used to evaluate the accuracy of the model.

Each image is preprocessed and resized to 128x128x3 before being fed into the network.

## Model Architecture

The DogFace recognition model is based on a **Siamese network** that learns to differentiate between dog faces. Each branch of the Siamese network uses a **CNN** (Convolutional Neural Network) to encode the images into feature vectors. The **triplet loss** function ensures that:

-   The **anchor** image (A) is closer to the **positive** image (P) (same dog) than it is to the **negative** image (N) (different dog) in the embedding space.

The model consists of:

-   **Convolutional layers**: Extract feature maps from the input image.
-   **Dense layers**: Output a fixed-length embedding.
-   **Triplet Loss**: Minimize the distance between similar dog faces and maximize the distance between different dog faces.

## Triplet Loss

![Triplets](https://i.ibb.co/mCzG66dS/dogs.png)


Triplet loss is the core component of the training process. The loss function ensures that the embeddings generated for images of the same dog are closer than the embeddings of images of different dogs. The loss is defined as:
$$
L(A,P,N)=max⁡(0,∥f(A)−f(P)∥2−∥f(A)−f(N)∥2+α)L(A, P, N) = \max(0, \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha)L(A,P,N)=max(0,∥f(A)−f(P)∥2−∥f(A)−f(N)∥2+α)
$$
Where:

-   AAA is the anchor image.
-   PPP is the positive image (same dog).
-   NNN is the negative image (different dog).
-   f(x)f(x)f(x) represents the embedding function (CNN).
-   α\alphaα is the margin between positive and negative pairs.

![By SachaDee](https://i.ibb.co/LhXmFQG/hugg.png)


## Ideal Use Case

**NGOs, Pet Insurance Companies and Public Administration**

This DogFace recognition system, powered by ONNX and TFLite models, is specifically designed to help **NGOs** and **public administrations** in their mission to care for and reunite lost dogs with their owners. The solution offers a highly efficient way to manage and track lost and found dogs through advanced dog face recognition, ensuring fast and accurate identification. Here's why this system is an ideal solution:

### 1. **NGOs Focused on Lost Dogs**

Organizations that work with lost or stray animals often face challenges when it comes to reuniting dogs with their owners. Traditional methods, such as relying on physical characteristics, microchips, or tags, can sometimes be inefficient or unavailable. Our model provides:

-   **Automated dog identification**: Through a simple image capture, NGOs can quickly determine whether a dog found on the street matches one of the dogs in their lost dog database.
-   **Mobile-friendly deployment**: The TFLite version of the model enables easy deployment on mobile devices, allowing rescue workers and volunteers to scan dogs in real-time using smartphones or other portable devices.
-   **1:N Matching:** The model can compare a newly found street dog against a database of lost dogs, making it easy to identify a potential match.

This streamlined identification process reduces the time required to reunite dogs with their owners and helps NGOs efficiently manage large volumes of data for lost and found dogs.

### 2. **Public Administration and Animal Control**

Government bodies responsible for animal welfare, especially in urban settings, can benefit from this solution by:

-   **Maintaining a centralized database** of lost and stray dogs. When a dog is brought into a shelter or is reported as found, its photo can be matched against existing records.
-   **Automating searches** when new street dogs are registered. By incorporating this model into their workflow, municipalities can reduce the time and resources needed for manual identification processes, and prioritize matching lost dogs with their owners faster.
-   **Cross-jurisdictional collaboration**: Public administrations can share databases across different regions, allowing for more efficient cross-referencing and identification in cases where dogs have crossed city or regional borders.

### 3. **Pet Insurance Companies**

Pet insurance companies face the challenge of verifying that a dog making a claim is indeed the one covered by the insurance policy. This DogFace recognition system provides an effective solution for fraud prevention and identity verification in the pet insurance industry by:

-   **Accurate dog identity verification**: When a pet owner makes an insurance claim, the company can use this model to verify that the dog involved in the claim is the same dog that was originally insured. This reduces the chances of fraudulent claims where a different dog might be presented to claim benefits.
-   **Easy integration into claim processes**: The model can be integrated into mobile apps or online platforms, allowing pet owners to upload a current image of their dog for verification. This makes the claim process smoother and more secure.
-   **Cross-device support**: With the TFLite version of the model, insurance companies can allow their field agents or veterinarians to perform real-time verification during house visits, examinations, or claim assessments, using just a smartphone or tablet.

By incorporating this technology into their workflows, pet insurance companies can ensure the authenticity of claims, protecting both the company and the policyholders from potential fraud. Additionally, it enhances the customer experience by speeding up the claim verification process.

### **Scalability and Accessibility**

-   **ONNX for large-scale deployments**: The ONNX version of the model ensures compatibility across a wide range of platforms, making it suitable for integration into larger public administration systems, such as those used for animal control or registration.
-   **TFLite for mobile solutions**: The lightweight TFLite model is optimized for devices with limited computational power, making it ideal for fieldwork conducted by animal control officers, veterinarians, and rescue volunteers.


### 4. **Improved Accuracy Over Time**

**NGOs**, **Pet Insurance Companies** and **public administrations** can benefit from continuously improving the system’s accuracy by adding new images of dogs to the database. As more data is collected, the model becomes increasingly effective at matching dogs, especially in difficult cases where a dog's appearance may have changed over time (e.g., due to grooming or injury).

By leveraging this dog face recognition model, organizations can significantly enhance their ability to identify lost dogs and reunite them with their families, while reducing costs, time, and human errors associated with manual identification.

## References

-   [DogFaceNet Dataset](https://github.com/DogFaceNet/dataset)
-   [GuillaumeMougeot DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet)
-   [Triplet Loss for Face Recognition](https://arxiv.org/abs/1503.03832)
-   [Siamese Networks](https://keras.io/examples/vision/siamese_network/)

