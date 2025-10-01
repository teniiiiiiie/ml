import tensorflow as tf
import tf2onnx
import onnx
import keras

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model

def convert_keras_to_onnx(model_path=None, img_size=(256, 256), num_classes=2, onnx_path="model.onnx"):
    if model_path and model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:
        model = get_model(img_size, num_classes)
    
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"Model converted to ONNX: {onnx_path}")
    return onnx_path

def convert_savedmodel_to_onnx(saved_model_dir, onnx_path="model.onnx"):
    model = tf.keras.models.load_model(saved_model_dir)
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"SavedModel converted to ONNX: {onnx_path}")

if __name__ == "__main__":
    convert_keras_to_onnx("model.h5", onnx_path="model.onnx")