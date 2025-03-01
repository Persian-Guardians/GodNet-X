import tensorflow as tf
import os
import json
import tensorflow_datasets as tfds

if tf.config.list_physical_devices('GPU'):
    print("GodNet-X Metal GPU Detected 🔥")
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("GodNet-X CPU Only")
    
os.environ["TF_DISABLE_MLIR"] = "1"  # TensorFlow legacy mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence unnecessary logs
tf.config.set_visible_devices([], 'GPU')
print("GodNet-X Fallback to CPU Mesh")

os.environ["TF_CONFIG"] = '{"cluster": {"chief": ["192.168.31.73:12345"],"worker": ["192.168.31.100:12345"]},"task": {"type": "chief","index": "0"}}'
    
strategy = tf.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: ', strategy.num_replicas_in_sync)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
tf_config = json.loads(os.environ["TF_CONFIG"])
NUM_WORKERS = len(tf_config["cluster"]["worker"]) + 1  # +1 برای chief

if tf_config["task"]["type"] == "worker":
    print(f"GodNet-X Node Type: Worker-{tf_config['task']['index']}")
else:
    print("GodNet-X Node Type: Chief")

# (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

(train_dataset, test_dataset), info = tfds.load('cifar10', with_info=True, as_supervised=True)

# پیش‌پردازش داده‌ها (در صورت نیاز)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)

# تقسیم داده‌ها بین نودها
train_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dataset = strategy.experimental_distribute_dataset(test_dataset)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

def prepare_for_model(dataset):
    return dataset.map(lambda x, y: (x, y))

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    train_dataset = prepare_for_model(train_dataset)
    test_dataset = prepare_for_model(test_dataset)
    
    model.fit(train_dataset, epochs=10)
    
