Vocal autoencoder
=================

This is a collection of experiments using raw audio with tensorflow.

The experiments are as follows:

# train_wnn.py

This is a wavelet neural network autoencoder.  It first converts signals to wavelets, then uses a hidden layer to compress.

# train_wnn_conv.py

Same as train_wnn.py, but uses convolutions in the hidden layers

# train_wnn_conv_lstm.py

Uses an LSTM inside the hidden layers.

Can be pretrained without the LSTM and then have the LSTM trained separately.

# train_wnn_conv_lstm_adversarial.py
Uses an adversarial pretrainer to (hopefully) find better features in the wavelets.  

# note
A lot of this is quickly done and experimental.  If you have trouble or need help running things, let us know.

