class EarlyStopper:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_loss = float('inf')
    
    def __call__(self, current_loss):
        if current_loss < self.min_loss - self.delta:
            self.min_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False