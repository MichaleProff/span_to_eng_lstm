from keras.models import Model, load_model

test_model = load_model('training_model.h5')
print(test_model.layers[0]._inbound_nodes)