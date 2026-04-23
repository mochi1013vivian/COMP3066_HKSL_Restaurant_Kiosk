> **Archived draft notice**
>
> This file is an older proposal draft and is **not the authoritative project definition anymore**.
> The active proposal direction is documented in `project_proposal_final.md`.
> This draft includes earlier ideas such as A-Z / 0-9 recognition and speech-recognition components that are no longer part of the core coursework scope.

COMP3066 Group Project Proposal 

Instructor: Prof. GUO Xiaoqing & Dr. ZHANG Ce 

Course code: COMP3066 

Course name: Health and Assistive Technology: Practicum 

  

  

Group 3 

Wong Hei Tung 23227311 

Liu Hiu Lam 23233494 

Cheung Sin Kay Vivian 23235845 

  

 

  

 

 

Project Proposal 

1.Introduction: 

It has been a challenge for deaf and hard-of-hearing persons to order food at restaurants. In most cases, the staff at the restaurant are not aware of the sign language, and the only option they can use in such a scenario is the drive-through speaker system. In such cases, there is a lot of miscommunication, and the order does not come out right. Moreover, the deaf and hard-of-hearing persons get a feeling of being excluded from the experience of going out to dine at a restaurant (McLeod, 2019).  

With the increasing popularity of digital menus and other such systems, it would be a golden opportunity to make the deaf and hard-of-hearing persons feel included in the experience of going out to dine at a restaurant. Past research has shown the potential of developing such systems for the deaf and hard-of-hearing persons. Such systems would be able to overcome the communication barriers of the deaf and hard-of-hearing persons. It would be a comfortable experience for them to dine at a restaurant. 

Sign language recognition technology is developing at a rapid rate. Real-time sign language recognition technology has achieved success in recognizing static signs such as fingers spelled words and numbers using hand tracking technology such as Mediapipe. Furthermore, the technology has achieved success in recognizing dynamic signs. It is stated that the accuracy of the technology is up to 90-98% (Dubey, 2025). Recently, the technology has been applied in practical fields such as restaurant ordering systems for particular sign languages (Al Khuzayem et al., 2024). 

Following these developments, in this project, a real-time food ordering assistant system based on sign language has been created. In this system, a standard webcam is utilized to detect hand gestures, and MediaPipe is used to detect specific hand features, which are then passed through a classification model trained on letters from A to Z, numbers from 0 to 9, and some specific dynamic signs related to food items, such as "burger," "water," etc., and some phrases such as "bill." These signs are then translated into corresponding characters and combined to form a sentence displaying what has been ordered on the screen, which can then be reviewed by the staff at the restaurant. The main objective of creating such systems is to help the deaf and hard of hearing order food independently and confidently, thus making them feel more included and accessible in society, especially in restaurants and other public areas. Most recently, these systems have been utilized in real-life situations, e.g., ordering systems for specific sign languages at restaurants (Al Khuzayem et al., 2024). 

 

2.Method: 

Dataset 

Two distinct datasets will be used for two main components of the system: Sign recognition and Speech recognition. 

For sign recognition, custom dataset of restaurant-related sign language vocabulary will be collected. This dataset will include essential food items (e.g., burger, pizza, water, coffee), quantities (e.g., one, two, three), and common ordering phrases (e.g., "I want," "please," "thank you"). The dataset will consist of images captured from multiple angles, under varying lighting conditions, and performed by multiple individuals to ensure stability. Raw images are also required as input for MediaPipe hand detection and landmark extraction e.g. 200 images per class. 

For speech recognition, audio dataset of restaurant ordering phrases and common spoken language such as “Excuse me” and “Thank you” will be involved. This is to make the system also understand verbal inputs from hearing individuals and facilitate translation as well. 

 

Feature Extraction and Hand Detection for Sign Recognition 

The raw image is a high-dimensional image that includes unnecessary data like background and arm position. Hence, a separate process to detect and extract features of the hand is required. We will use Mediapipe Hands to extract targeted features for our project. 

For each image, the pipeline will try to find out if a hand is visible. If it is, it will try to find out where it is and determine 21 key points on it in 3D coordinates (x, y, z). These 21 key points will be on the palm of the hand, the fingertips, and the joints. This will give us a succinct representation of the hand, accounting for size, skin color, and background. 

 

Audio Processing and Feature Extraction for Speech Recognition 

For the component of Speech-to-Text, raw audio waveforms of responses from the restaurant staff will be processed and relevant features will be extracted. Mel Frequency Cepstral Coefficients will be used as the primary feature set for this component, as they have proven effective in representing human speech signals in an efficient manner. 

 

Classifier (Sign Recognition) 

The extracted 21 hand landmarks (resulting in a 63-dimensional feature vector, 21 points * 3 coordinates) will be our input for the classifier. We will use two different classifiers in our experiments. First, we will use a Multiclass SVM classifier. To tackle the complex and non-linear relationships between the landmark positions for different signs, we will use a simple neural network classifier. We will compare our classifier performance to this classifier. The model will classify the static hand pose into the corresponding letter (A-Z) or number (1-9). The model will classify the static hand pose or dynamic gesture sequence into the corresponding restaurant vocabulary item ("burger," "water," "one," etc.). 

 

Classifier (Speech Recognition) 

The extracted MFCC features will be fed into a sequence-to-sequence model, such as a recurrent neural network (RNN) with Long Short-Term Memory (LSTM) units or a Connectionist Temporal Classification (CTC) model. This model will learn the mapping between acoustic features and corresponding text outputs, enabling the translation of spoken phrases from restaurant staff (e.g., "What would you like to order?") into text for the deaf user. 

 

Evaluation 

Accuracy is the primary evaluation metric in our project. Our algorithm will be evaluated over several experiment trials using a held-out test set. In each trial, 70% of the data will be used for training, and the remaining 30% for testing. Furthermore, the system's performance will be qualitatively evaluated on new, real-time images captured via a webcam. 

 

3.Goal 

The main objective of this project is to design a real-time food ordering assistant using sign language, which should operate on a normal laptop and webcam. The system should recognize static signs for letters from A to Z and numbers from 0 to 9, as well as some dynamic signs related to ordering food at a restaurant, such as "burger," "water," "bill," "thank you," etc. In addition, for static sign language recognition, we expect to achieve an accuracy of 90-95% on a held-out test set under controlled circumstances. 

These signs will be mapped to words or tokens and combined to create a simple text-based food order. The text-based food order will be displayed on the screen for the restaurant staff to review. The successful implementation of the image-based ordering system from the recognition of a single sign to the creation of an order on the screen is the main objective of the project. As an extended objective of the project, we would like to work on a basic video-based system where predictions are aggregated over a short sequence of frames to make the sign recognition process smoother. The system is designed to enable deaf and hard-of-hearing customers to order food independently. 

 

4.Timeframe 

Description of Work 

Start and End dates 

Foundation & Setup 

Friday, Week 10 - Friday, Week 11 

Hand Detection Implementation 

Saturday, Week 11 - Wednesday, Week 12 

Sign and Speech Recognition Model 

Thursday, Week 12 - Wednesday, Week 13 

Translation & Simple Sentences 

Thursday, Week 12 - Wednesday, Week 13 

Documentation & Presentation 

Thursday, Week 13 - Presentation Day, Week 14 

 

 

 

References 

Al Khuzayem, L. A., Shukri, S., Alghamdi, S., Alghamdi, R., & Alotaibi, O. (2024). Efhamni: A deep learning-based Saudi Sign Language recognition application. Sensors, 24(10), Article 3112. https://doi.org/10.3390/s24103112 

Al Zadjali, S., & Khan, S. M. H. (2019). Application of sign language in designing restaurant’s menu for deaf people. International Journal of Recent Technology and Engineering, 7(5S), 267–271. https://www.ijrte.org/wp-content/uploads/papers/v7i5s/ES2153017519.pdf 

Dubey, A. (2025). Real-time sign language recognition using MediaPipe and deep learning approaches: A mobile application integration. Journal of Emerging Technologies and Innovative Research, 12(5), 795–800. https://www.jetir.org/papers/JETIR2505309.pdf 

McLeod, R. (2019). Deaf and hard of hearing accessibility at drive-through restaurants (Undergraduate honors thesis). Western Michigan University. https://scholarworks.wmich.edu/honors_theses/3115 

 

 

 