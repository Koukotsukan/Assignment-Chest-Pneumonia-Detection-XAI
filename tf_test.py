if __name__ == '__main__':

    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.models import Model
    import tensorflow as tf
    from models.model_loader import load_vgg16

    model = load_vgg16()
    for layer in model.layers:
        print(f"Layer: {layer.name}")
