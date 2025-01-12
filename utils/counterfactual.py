from scipy.ndimage import gaussian_filter
import numpy as np
import tensorflow as tf


def simulate_counterfactual(image_array, model, target_proba, base_alpha=0.95, step_alpha=0.05, steps=300):
    """
    基于模型预测生成反事实图像，并根据预测动态调整对比度。
    """
    def adjust_contrast(image_array, alpha):
        # 对比度调整，保持形状一致
        mean_pixel = tf.reduce_mean(image_array, axis=(1, 2, 3), keepdims=True)
        return alpha * image_array + (1 - alpha) * mean_pixel

    # 确保输入图像为 float32 类型并有批量维度
    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if image_array.ndim == 3:
        image_array = tf.expand_dims(image_array, axis=0)
    print("Initial Input Image Shape:", image_array.shape)

    # 转换为可优化的张量
    image_tensor = tf.Variable(image_array, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    for step in range(steps):
        print(f"Step {step}: Starting, Image Tensor Shape: {image_tensor.shape}")

        with tf.GradientTape() as tape:
            tape.watch(image_tensor)

            # 模型预测
            predictions = model(image_tensor)
            print(f"Step {step}: Predictions Shape: {predictions.shape}")

            # 损失函数
            classification_loss = tf.abs(predictions[0, 0] - target_proba)
            regularization_loss = tf.reduce_mean(tf.square(image_tensor - image_array))
            loss = classification_loss + 0.5 * regularization_loss
            print(f"Step {step}: Loss: {loss.numpy()}")

        # 计算梯度并检查梯度
        gradients = tape.gradient(loss, image_tensor)
        print(f"Step {step}: Gradients Shape: {gradients.shape}")
        if tf.reduce_any(tf.math.is_nan(gradients)):
            print("NaN detected in gradients! Stopping optimization.")
            break
        if tf.reduce_any(tf.math.is_inf(gradients)):
            print("Inf detected in gradients! Stopping optimization.")
            break

        # 应用梯度更新
        optimizer.apply_gradients([(gradients, image_tensor)])
        print(f"Step {step}: After Gradient Update, Image Tensor Shape: {image_tensor.shape}")

        # 动态调整对比度
        if step % 50 == 0:
            current_proba = predictions[0, 0].numpy()
            if current_proba > target_proba:
                alpha = base_alpha - step_alpha * (current_proba - target_proba)
                # alpha = max(alpha, 0.5)  # 防止对比度过低
            else:
                alpha = base_alpha + step_alpha * (target_proba - current_proba)
                # alpha = min(alpha, 1.5)  # 防止对比度过高

            # 调整对比度并重新分配
            adjusted_image = adjust_contrast(image_tensor, alpha=alpha)
            image_tensor.assign(tf.clip_by_value(adjusted_image, 0.0, 1.0))
            print(f"Step {step}: After Contrast Adjustment, Image Tensor Shape: {image_tensor.shape}, Alpha: {alpha:.2f}")

        # 限制像素值范围
        image_tensor.assign(tf.clip_by_value(image_tensor, 0.0, 1.0))
        print(f"Step {step}: After Clipping, Image Tensor Shape: {image_tensor.shape}")

    # 最终检查形状并返回
    final_image = image_tensor.numpy()
    print("Final Image Tensor Shape:", final_image.shape)
    if final_image.ndim == 3:
        final_image = np.expand_dims(final_image, axis=0)
    print("Final Counterfactual Image Shape (With Batch):", final_image.shape)
    return final_image


# 预测时也需要保证输入形状
def predict_image(image_array, model):
    # 添加批量维度
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)

    # 确保是 4D 张量
    if len(image_array.shape) != 4:
        raise ValueError(f"Input image must have shape (batch_size, height, width, channels). Got: {image_array.shape}")

    # 执行预测
    predictions = model.predict(image_array)
    return predictions


def smooth_image(image_array, sigma):
    return gaussian_filter(image_array, sigma=sigma)




