# =========================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =========================================================
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# =========================================================
# STEP 2: SYNTHETIC MEDICAL IMAGE DATASET GENERATION
# =========================================================
def generate_synthetic_data(num_images=200, img_size=128):
    images = []
    masks = []

    for _ in range(num_images):
        image = np.zeros((img_size, img_size), dtype=np.uint8)
        mask  = np.zeros((img_size, img_size), dtype=np.uint8)

        num_cells = np.random.randint(3, 8)
        for _ in range(num_cells):
            x, y = np.random.randint(20, img_size - 20, size=2)
            r    = np.random.randint(8, 15)

            cv2.circle(image, (x, y), r, np.random.randint(120, 255), -1)
            cv2.circle(mask,  (x, y), r, 255, -1)

        images.append(image)
        masks.append(mask)

    images = np.array(images) / 255.0
    masks  = np.array(masks)  / 255.0

    return images, masks


X, y = generate_synthetic_data()

print("Images shape:", X.shape)
print("Masks shape:", y.shape)


# =========================================================
# STEP 3: TRAIN–VALIDATION SPLIT
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
y_train = y_train[..., np.newaxis]
y_val   = y_val[..., np.newaxis]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# =========================================================
# STEP 4: DEFINE DICE LOSS AND IoU METRIC
# =========================================================
def dice_loss(y_true, y_pred):
    smooth       = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_metric(y_true, y_pred):
    y_pred       = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)


# =========================================================
# STEP 5: DEFINE U-NET BUILDING BLOCK
# =========================================================
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# =========================================================
# STEP 6: BUILD U-NET MODEL FROM SCRATCH
# =========================================================
def unet(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    bn = conv_block(p3, 512)
    bn = Dropout(0.5)(bn)

    # Decoder
    u1 = Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u1 = concatenate([u1, c3])
    c4 = conv_block(u1, 256)

    u2 = Conv2DTranspose(128, 2, strides=2, padding="same")(c4)
    u2 = concatenate([u2, c2])
    c5 = conv_block(u2, 128)

    u3 = Conv2DTranspose(64, 2, strides=2, padding="same")(c5)
    u3 = concatenate([u3, c1])
    c6 = conv_block(u3, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(c6)

    return Model(inputs, outputs)


model = unet()
model.summary()


# =========================================================
# STEP 7: TRAIN MODEL WITH ADAM OPTIMIZER
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,
    metrics=[iou_metric]
)

history_adam = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)


# =========================================================
# STEP 8: TRAIN MODEL WITH SGD + MOMENTUM
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    loss=dice_loss,
    metrics=[iou_metric]
)

history_sgd = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)


# =========================================================
# STEP 9: PLOT IoU CURVES (TRACKING PERFORMANCE)
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(history_adam.history['val_iou_metric'], label='Adam Validation IoU')
plt.plot(history_sgd.history['val_iou_metric'],  label='SGD Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('Adam vs SGD - Validation IoU')
plt.legend()
plt.grid()
plt.show()


# =========================================================
# STEP 10: FINAL METRICS CALCULATION
# =========================================================
def dice_coefficient(y_true, y_pred):
    y_pred       = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6
    )


predictions = model.predict(X_val)

final_dice = dice_coefficient(
    tf.convert_to_tensor(y_val),
    tf.convert_to_tensor(predictions)
)

final_iou = iou_metric(
    tf.convert_to_tensor(y_val),
    tf.convert_to_tensor(predictions)
)

print("Final Validation Dice Coefficient:", float(final_dice))
print("Final Validation IoU:", float(final_iou))


# =========================================================
# STEP 11: QUALITATIVE SEGMENTATION RESULT
# =========================================================
idx       = 0
pred_mask = model.predict(X_val[idx:idx + 1])

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(X_val[idx].squeeze(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(y_val[idx].squeeze(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred_mask.squeeze(), cmap='gray')

plt.show()
