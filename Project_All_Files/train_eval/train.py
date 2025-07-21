from config import OPTIMIZER_VAL, LOSS_VAL 
def train_model(model, X_train, y_train, epochs, batch_size):
    model.compile(optimizer=OPTIMIZER_VAL, loss=LOSS_VAL , metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model