"""Python file to instantite the model and the transform that goes with it."""
from model import Net
from data import data_transforms, data_transforms_train


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()
        self.transform_train = self.init_transform_train()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")
        
    def init_transform_train(self):
        if self.model_name == "basic_cnn":
            return data_transforms_train
        else:
            raise NotImplementedError("Transform train not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform, self.transform_train
