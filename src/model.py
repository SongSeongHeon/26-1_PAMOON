
import tensorflow as tf

from src.config import BEAT_LEN, EMBED_DIM


# =========================================================
# Common blocks
# =========================================================
def residual_block_1d(x, filters, kernel_size=5, downsample=False, name_prefix="res"):
    stride = 2 if downsample else 1
    shortcut = x

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1",
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu1")(y)

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2",
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(y)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name_prefix}_proj",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    out = tf.keras.layers.Add(name=f"{name_prefix}_add")([y, shortcut])
    out = tf.keras.layers.Activation("relu", name=f"{name_prefix}_out")(out)
    return out


def conv_bn_relu(
    x,
    filters,
    kernel_size,
    strides=1,
    dropout_rate=None,
    name_prefix="convblk",
):
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv",
    )(x)

    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu")(x)

    if dropout_rate is not None and dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)

    return x


def make_embedding_head(x, num_classes=90, embed_dim=EMBED_DIM):

    x = tf.keras.layers.Dropout(0.35, name="dropout_pre_embed")(x)

    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_256_bn")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_mid")(x)

    embedding = tf.keras.layers.Dense(
        embed_dim,
        activation=None,
        name="embedding_256",
    )(x)

    norm_embedding = tf.keras.layers.UnitNormalization(
    axis=1,
    name="embedding_l2norm"
)(embedding)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(norm_embedding)

    return outputs


# =========================================================
# Model 1: ResNet1D baseline (keep as-is)
# =========================================================
def build_resnet1d_model(input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    inputs = tf.keras.layers.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(inputs)

    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="stem_pool")(x)

    x = residual_block_1d(x, 32, kernel_size=5, downsample=False, name_prefix="res1")
    x = residual_block_1d(x, 64, kernel_size=5, downsample=True, name_prefix="res2")
    x = residual_block_1d(x, 128, kernel_size=3, downsample=True, name_prefix="res3")

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="global_concat")([avg_pool, max_pool])

    outputs = make_embedding_head(
        x=x,
        num_classes=num_classes,
        embed_dim=embed_dim,
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ECG_ResNet1D_Model")
    return model


# =========================================================
# Model 2: Tuned Plain CNN1D
# =========================================================
def build_plain_cnn1d_model(input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    inputs = tf.keras.layers.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(inputs)

    x = conv_bn_relu(x, 32, 7, strides=1, dropout_rate=0.05, name_prefix="block1")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="block1_pool")(x)

    x = conv_bn_relu(x, 64, 5, strides=1, dropout_rate=0.10, name_prefix="block2")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="block2_pool")(x)

    x = conv_bn_relu(x, 128, 5, strides=1, dropout_rate=0.12, name_prefix="block3")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="block3_pool")(x)

    x = conv_bn_relu(x, 256, 3, strides=1, dropout_rate=0.15, name_prefix="block4")
    x = conv_bn_relu(x, 256, 3, strides=1, dropout_rate=0.15, name_prefix="block5")

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="global_concat")([avg_pool, max_pool])

    outputs = make_embedding_head(
        x=x,
        num_classes=num_classes,
        embed_dim=embed_dim,
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ECG_PlainCNN1D_Model")
    return model


# =========================================================
# Model 3: Tuned BiLSTM
# =========================================================
def build_bilstm_model(input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    inputs = tf.keras.layers.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(inputs)

    # stronger CNN front-end before recurrent modeling
    x = conv_bn_relu(x, 32, 7, strides=1, dropout_rate=0.05, name_prefix="front1")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="front1_pool")(x)

    x = conv_bn_relu(x, 64, 5, strides=1, dropout_rate=0.08, name_prefix="front2")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="front2_pool")(x)

    x = conv_bn_relu(x, 96, 3, strides=1, dropout_rate=0.10, name_prefix="front3")

    # lighter / better balanced recurrent stack
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.15),
        name="bilstm1",
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.15),
        name="bilstm2",
    )(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="global_concat")([avg_pool, max_pool])

    outputs = make_embedding_head(
        x=x,
        num_classes=num_classes,
        embed_dim=embed_dim,
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ECG_BiLSTM_Model")
    return model


# =========================================================
# Model 4: Tuned CNN + BiLSTM
# =========================================================
def build_cnn_bilstm_model(input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    inputs = tf.keras.layers.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(inputs)

    x = conv_bn_relu(x, 32, 7, strides=1, dropout_rate=0.05, name_prefix="cnn1")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="cnn1_pool")(x)

    x = conv_bn_relu(x, 64, 5, strides=1, dropout_rate=0.08, name_prefix="cnn2")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="cnn2_pool")(x)

    x = conv_bn_relu(x, 128, 3, strides=1, dropout_rate=0.10, name_prefix="cnn3")
    x = conv_bn_relu(x, 128, 3, strides=1, dropout_rate=0.10, name_prefix="cnn4")

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True, dropout=0.15),
        name="bilstm1",
    )(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    x = tf.keras.layers.Concatenate(name="global_concat")([avg_pool, max_pool])

    outputs = make_embedding_head(
        x=x,
        num_classes=num_classes,
        embed_dim=embed_dim,
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ECG_CNN_BiLSTM_Model")
    return model


# =========================================================
# Backward compatibility
# =========================================================
def build_record_generalization_model(input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    return build_resnet1d_model(
        input_shape=input_shape,
        num_classes=num_classes,
        embed_dim=embed_dim,
    )


# =========================================================
# Factory
# =========================================================
def build_model(model_name="resnet1d", input_shape=(BEAT_LEN, 1), num_classes=90, embed_dim=EMBED_DIM):
    model_name = model_name.lower().strip()

    if model_name in ["resnet", "resnet1d", "record_generalization"]:
        return build_resnet1d_model(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    if model_name in ["plain_cnn", "plain_cnn1d", "cnn", "cnn1d"]:
        return build_plain_cnn1d_model(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    if model_name in ["bilstm", "lstm"]:
        return build_bilstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    if model_name in ["cnn_bilstm", "cnn+lstm", "cnn_lstm", "hybrid"]:
        return build_cnn_bilstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Available: ['resnet1d', 'plain_cnn1d', 'bilstm', 'cnn_bilstm']"
    )
