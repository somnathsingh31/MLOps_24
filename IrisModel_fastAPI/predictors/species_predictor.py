import joblib
from sklearn.preprocessing import StandardScaler

class IrisClassifier:
    def __init__(self, model_path, data):
        self.model_path = model_path
        self.data = data

    def _load_model(self):
        return joblib.load(self.model_path)

    def preprocess(self, data):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data)
        return x_scaled


    def predictor(self):
        scaled_data = self.preprocess(self.data)
        model = self._load_model()
        return model.predict(scaled_data)[0]