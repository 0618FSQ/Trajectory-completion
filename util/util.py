import torch



class Util:
    @staticmethod
    def freeze(model):
        for k, v in model.named_parameters():
            v.requires_grad = False

    @staticmethod
    def unfreeze(model):
        for k, v in model.named_parameters():
            v.requires_grad = True