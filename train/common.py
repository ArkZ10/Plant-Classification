"""Shared compile/fit/save logic used by every train_model*.py script."""

import json
import os


def train_and_save(model, name, train_generator, validation_generator,
                   epochs, ckpt_dir, history_dir):
    """Compile with Adam + categorical crossentropy, fit, and persist results."""
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, f"{name}.keras"))
    with open(os.path.join(history_dir, f"{name}.json"), "w") as f:
        json.dump(history.history, f)
    print(f"Saved {ckpt_dir}/{name}.keras and {history_dir}/{name}.json")

    return history
