class ExponentialMovingAverage:
    def __init__(self, beta):
        self.beta = beta
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.beta * self.shadow[name] + (1.0 - self.beta) * param.data
                self.shadow[name] = new_average.clone()
    def apply_shadow(self, model):
        msg = model.load_state_dict(self.shadow, strict=True)
        print('apply_shadow: '+str(msg))