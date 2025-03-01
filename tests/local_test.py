import tensorflow as tf
import numpy as np

class PersianViT(tf.keras.Model):
    def __init__(self, hidden_dim=128, num_classes=10):
        super(PersianViT, self).__init__()
        self.conv = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

def run_local():
    print("[INFO] GodNet-X Local Test Starting...")
    model = PersianViT()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0

    print("[INFO] Training Started...")
    model.fit(x_train, y_train, epochs=1, batch_size=32)

if __name__ == "__main__":
    run_local()
    print("[SUCCESS] GodNet-X Local Test Success")