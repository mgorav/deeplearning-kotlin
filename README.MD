A  multilayered (MLP) applied to digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).

This example uses two input layers and one hidden layer.

The first input layer has input dimension of numRows*numColumns where these variables indicate the
number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
(relu) activation function. The weights for this layer are initialized by using Xavier initialization
(https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
to avoid having a steep learning curve. This layer sends 500 output signals to the second layer.

The second input layer has input dimension of 500. This layer also uses a rectified linear unit
(relu) activation function. The weights for this layer are also initialized by using Xavier initialization
(https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
to avoid having a steep learning curve. This layer sends 100 output signals to the hidden layer.

The hidden layer has input dimensions of 100. These are fed from the second input layer. The weights
for this layer is also initialized using Xavier initialization. The activation function for this
layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
add up to 1. The highest of these normalized values is picked as the predicted class.


NOTE: WIP, gradle/pom dependencies are missing