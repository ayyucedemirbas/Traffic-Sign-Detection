from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
class SignNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		
		inputShape = (height, width, depth)
		chanDim = -1
        model = Sequential([
            Conv2D(8, (5, 5), padding="same",activation='relu',input_shape=inputShape),
            BatchNormalization(axis=chanDim),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=chanDim),
            Conv2D(16, (3, 3), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=chanDim),
            Conv2D(32, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=chanDim),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128,activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Flatten(),
            Dense(128,activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(classes, activation='softmax')
    ])
		return model
