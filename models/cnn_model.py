from keras.layers import Conv1D, Embedding, Input, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model
from keras.layers.merge import concatenate


def build_model(vocab_size=10000, max_length=350, kernel_sizes = (3,5,7), num_filters = (48,48,48), pool_size=2,dropout_rate=0.5, embedding_size=100):
    if len(kernel_sizes) < 1:
        raise ValueError("At least one input stage is required.")
    if len(kernel_sizes) != len(num_filters):
        raise ValueError("Number of filter sizes needs to match number of input stages.")
	
    inputs = []
    input_stages = []
	
    for kernel_size, filters in zip(kernel_sizes, num_filters):
        input_ = Input(shape=(max_length,))
        embedding = Embedding(vocab_size, embedding_size)(input_)
        conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(embedding)
        drop = Dropout(dropout_rate)(conv)
        pool = MaxPooling1D(pool_size=pool_size)(drop)
        flat = Flatten()(pool)
    
        inputs.append(input_)
        input_stages.append(flat)
	
    merged = concatenate(input_stages)
	
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(2, activation='softmax')(dense1)
    model_cnn = Model(inputs=inputs, outputs=outputs)

    return model_cnn

if __name__ == "__main__":
    model = build_model()
    print(model.summary())




