# -*- coding: utf-8 -*-
"""Helpers NN cho notebook single-route (không phụ thuộc Streamlit)."""
from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Input, Model, layers


def build_demand_model(
    n_routes: int,
    n_num_features: int,
    *,
    use_route_embedding: bool = False,
    hidden: tuple[int, ...] = (32, 16),
    dropout: float = 0.30,
) -> Model:
    """NN dự báo residual log demand."""
    inp_num = Input(shape=(n_num_features,), name="num_features")
    x = inp_num
    inputs: list[Any] = [inp_num]

    if use_route_embedding and n_routes > 1:
        inp_route = Input(shape=(1,), name="route_idx", dtype="int32")
        route_emb = layers.Embedding(
            n_routes,
            4,
            name="route_emb",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-3),
        )(inp_route)
        route_emb = layers.Flatten()(route_emb)
        x = layers.Concatenate()([route_emb, inp_num])
        inputs = [inp_route, inp_num]

    for i, units in enumerate(hidden):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(5e-3),
            name=f"dense_{i}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    out = layers.Dense(1, name="residual_log_demand", kernel_initializer="zeros")(x)
    model = Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return model
