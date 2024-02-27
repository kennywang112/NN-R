# Neural-Network

To enhance my understanding of neural networks and their variants, I am constructing basic neural networks for this project without relying on external packages. I will continually update and list the lessons learned below.

## Model

1.  Neural Network
2.  Recurrent Neural Network
3.  Convolutional Neural Network
4.  Long Short-Term Memory
5.  Gated Recurrent Unit

## Activation Function use
| Type                         | last activation | loss function           |
|------------------------------|-----------------|-------------------------|
| binary                       | sigmoid         | binary_crossentropy     |
| multi, single classification | softmax         | Text                    |
| multi, multi classification  | sigmoid         | binary_crossentropy     |
| regression any               | None            | mse                     |
| regression 0\~1              | sigmoid         | mse/binary_crossentropy |
