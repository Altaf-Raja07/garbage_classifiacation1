import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

model = None

output_class = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

data = {
    "cardboard": [
        "Cardboard recycling helps conserve resources and reduce landfill waste. Most cardboard products, like boxes and packaging, are made from paper pulp, which can be recycled multiple times. Flatten the cardboard and keep it dry before recycling.",
        "zO3jFKiqmHo", "oKFOqMZmuA8"
    ],
    "glass": [
        "Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity. Separate glass by color (clear, green, brown) if required. Ensure items are clean and free from contamination before recycling.",
        "bYVih298o1Y", "6R8YObQbE88"
    ],
    "metal": [
        "Metal recycling reduces the need for mining and saves energy. Items like aluminum cans, tins, and foil can be recycled. Clean and dry them before placing in recycling bins.",
        "qAGCI0-pQ3E", "rgEEXhbar3A"
    ],
    "paper": [
        "Paper recycling reduces the demand for virgin paper and saves trees. Office paper, newspapers, and magazines are recyclable. Avoid recycling dirty or food-stained paper.",
        "jAqVxsEgWIM", "xhW0RTg8kRI"
    ],
    "plastic": [
        "Plastic recycling prevents pollution and conserves oil. Look for recyclable symbols on items like bottles and containers. Rinse and dry plastics before recycling.",
        "rYwBL_6hB2I", "I_fUpP-hq3A"
    ],
    "trash": [
        "Items classified as trash cannot be recycled. These include contaminated materials, mixed materials, or non-recyclables. Try to reduce the use of disposable items and dispose of trash responsibly.",
        "NhF4pXBNfq8", "8fFJOzXxB54"
    ]
}

def load_artifacts():
    global model
    model = tf.keras.models.load_model("model/garbage_model2.h5")

def classify_waste(image_path):
    global model, output_class
    img = image.load_img(image_path, target_size=(260, 260))  # ✅ Correct image size
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # ✅ EfficientNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    predicted_value = output_class[pred_index]

    return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2]
