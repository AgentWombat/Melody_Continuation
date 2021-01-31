# Melody_Continuation
Building a neural network with Tensorflow/Keras to continue melodies. MIDI format is used for music data.

In its current state, the data-parsing is complete and a basic model can be trained. The trained models typically predict uninteresting melodies, so it is far from finished.

When this Spring semester comes to an end, I hope to work more towards model optimization (eg. implementing a custom loss function) and data creation (I currently do not have enough training data).


#Files
THE MIDO LIBRARY IS REQUIRED FOR CODE TO FUNCTION. It can be installed via pip: "pip install mido"

The 'TestingModels' notebook is where the model is build and the main script is.

'midiToArray.py' holds many functions used for data parsing.

'data3.mid' is the data currently being used to train models which I created manually. It has 408 measures arraged in two-measure pairs: one measure of a setup and one measure of a resolution. The hope in that in using this to train models, the models might learn to compose endings spesifically.

