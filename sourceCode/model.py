from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 lớp đầu ra cho mèo, chim và chó

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Tạo mô hình
cnn_model = create_cnn_model()

# Sử dụng ImageDataGenerator để tải dữ liệu với tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/training_set',  
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'  
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/test_set',  
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

# Huấn luyện mô hình
cnn_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=40  # Có thể điều chỉnh số lượng epoch
)

# Lưu mô hình
cnn_model.save('saved_model/cat_bird_dog_classifier_v2.h5')  # Đổi tên file cho phù hợp với mô hình mới
