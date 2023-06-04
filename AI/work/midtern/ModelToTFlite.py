import tensorflow as tf
# Convert the trained TensorFlow model to the TensorFlow Lite (TFLite) format
converter = tf.lite.TFLiteConverter.from_saved_model('DNNFaceModel')
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
tflite_model = converter.convert()

# Save the TFLite model to disk
with open('DNNFaceModel.tflite', 'wb') as f:
    f.write(tflite_model)

# open('models/tflite/DNNmodel2.tflite', 'wb').write(tflite_model)