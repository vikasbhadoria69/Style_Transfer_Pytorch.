# Image Style Transfer by using Convolutional Neural Network.
* In this project, I have demonstrated how to use deep learning to compose images in the style
of another image . This is known as ‘neural style transfer’. This is a technique that produces
new images of high perceptual quality that combine the content of an arbitrary photograph
with the appearance of numerous well known artworks.
* A base input image is taken, a content image that has to be matched, and the style image that 
has to be matched. Then transforming the base input image by minimizing the content and
style distances (losses) with backpropagation, created a new image that matches the content of
the content image and the style of the style image.

#### Below are the content image and style image.

![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/City.jpg)![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/StarryNight.jpg)


## Code and Resources Used
**Python Version:** 3.7
**Packages:** pandas, numpy, matplotlib, vgg19, TransferLearning, PyTorch Framework. 

## Deep image representations
The project makes use of _‘transfer learning’_ technique which allows us to use the parameters
of a deep and highly trained model which has been extensively trained on huge datasets. This
task is otherwise very daunting because of the limited processing power. The model used in
this project is the **VGG-19** network model, which was trained to perform object recognition
and localisation. The model is publicly available and can be explored. VGG19 is trained on 14
million different images and is able to classify 1000 different object categories. This project
uses feature space provided by a normalised version of the 16 convolutional and 5 pooling
layers of the 19-layer VGG network. We use the 16 layer feature extraction part of VGG-19
and freeze its parameters so as to use the same weights as the model has achieved after
extensive training. There is no use of any fully connected layer of VGG19 as for this project
only feature extractions of an image are required, not the classification part.

## Content and Style representations
In order to get both the content and style representations of our image, some intermediate
layers within the model can be used. Intermediate layers represent feature maps that become
increasingly higher ordered as one goes deeper. These intermediate layers of VGG-19 are
necessary to define the representation of content and style from the images. For an input
image, the idea is to try and match the corresponding style and content target
representations at these intermediate layers

## Content Loss:
Content loss definition is actually quite simple. The network is passed with both the desired
content image and the base input image that needs to be transformed. This will return the
intermediate layer outputs from the model. Then simply take the euclidean distance between
the two intermediate representations of those images.

## Style Loss:
Computing style loss follows the same principle, this time feeding the network with base
input image and the style image. Instead of comparing the raw intermediate outputs of the
base input image and the style image, the _Gram matrices_ of the two outputs are compared.

## Run Gradient Descent
For this project the **‘Adam’** optimizer is used in order to minimize the loss. Iteratively the
output image is updated such that it minimizes our loss. The weights associated with the
model network are not changed, but instead the input image is trained to minimize loss.

## Workflow for this project

![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/Style_Transfer.jpg)

## Outputs

The target image has been itterated 9000 times through vgg19 mode in order to reduce the loss and update the params and to get desired style transfer.
##### Output target image after 1000 itterations
![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/style_feature1.png)

##### Output target image after 3000 itterations
![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/style_feature2.png)

##### Output target image after 6000 itterations
![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/style_feature3.png)

##### Output target image after 9000 itterations
![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/style_feature4.png)

##### Final Output comparison: 
Below, the first image is the content image followed by style image and the final image is the
output image.
It can be seen how the final output image has been created using the content of the first
image and style of the second.

![alt text](https://github.com/vikasbhadoria69/Style_Transfer_Pytorch./blob/master/Images/Style_image.png)


