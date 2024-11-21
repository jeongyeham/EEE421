import keras
import pandas as pd
import numpy as np

from keras import Model
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from pygments.lexers import go
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from matplotlib import pyplot as plt


class EnergyModel(Model):
    def __init__(self):
        super().__init__()
        self.features = None
        self.target = None
        self.dense_1 = Dense(512, activation='relu')
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(512, activation='relu')
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(128, activation='relu')
        self.dropout_3 = Dropout(0.2)
        self.dense_4 = Dense(1, activation='linear')
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_3(x)
        x = self.dropout_3(x, training=training)
        return self.dense_4(x)

    def build(self, input_shape):
        super().build(input_shape)

    def preprocess_training_data(self, file_path='./Training data.csv'):
        data = pd.read_csv(file_path).drop(columns=['Index'])
        self.target = data['ENERGY_CONSUMPTION_CURRENT']
        self.features = data.drop(columns=['ENERGY_CONSUMPTION_CURRENT'])
        features_scaled = self.feature_scaler.fit_transform(self.features)
        self.energy_model_feature_scaler_save()
        target_scaled = self.target_scaler.fit_transform(self.target.values.reshape(-1, 1))
        self.energy_model_target_scaler_save()
        return train_test_split(features_scaled, target_scaled, test_size=0.1)

    def energy_model_compile(self):
        super().compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    def energy_model_train(self, feature_data, target_data):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)
        history = super().fit(feature_data, target_data, batch_size=64, epochs=1000, validation_split=0.2,
                              callbacks=[early_stopping])
        return history

    def energy_model_save(self, model_path='./model.keras', overwrite=True):
        super().save(model_path, overwrite=overwrite)

    def energy_model_feature_scaler_save(self, feature_scaler_path='./feature_scaler.pkl'):
        dump(self.feature_scaler, feature_scaler_path)

    def energy_model_target_scaler_save(self, target_scaler_path='./target_scaler.pkl'):
        dump(self.target_scaler, target_scaler_path)

    def energy_model_feature_scaler_load(self, feature_scaler_path='./feature_scaler.pkl'):
        self.feature_scaler = load(feature_scaler_path)

    def energy_model_target_scaler_load(self, target_scaler_path='./target_scaler.pkl'):
        self.target_scaler = load(target_scaler_path)

    def energy_model_load(self, model_path='./model.keras'):
        return keras.models.load_model(model_path)

    def energy_model_evaluate(self, feature_data, target_data):
        test_loss, test_mae = super().evaluate(feature_data, target_data)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    def energy_model_predict(self, new_data):
        self.energy_model_feature_scaler_load()
        new_data_scaled = self.feature_scaler.fit_transform(new_data)
        predictions_scaled = super().predict(new_data_scaled)
        self.energy_model_target_scaler_load()
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions

    def sensitivity_analysis(self, feature_data, feature_name):
        baseline = super().predict(feature_data)
        results = {}
        for i, feature in enumerate(feature_name):
            original_value = feature_data[:, i]
            feature_data[:, i] = original_value * 1.1
            new_prediction = super(EnergyModel, self).predict(feature_data)
            change = np.mean(new_prediction - baseline)
            results[feature] = change
            feature_data[:, i] = original_value
        return results


def model_training():
    model = EnergyModel()
    X_train, X_test, y_train, y_test = model.preprocess_training_data()
    model.energy_model_compile()
    model.energy_model_train(X_train, y_train)
    model.energy_model_evaluate(X_test, y_test)
    feature_names = ['POSTCODE', 'CURRENT_ENERGY_EFFICIENCY', 'PROPERTY_TYPE', 'BUILT_FORM', 'TRANSACTION_TYPE',
                     'TOTAL_FLOOR_AREA', 'ENERGY_TARIFF', 'FLOOR_LEVEL', 'GLAZED_TYPE', 'GLAZED_AREA',
                     'NUMBER_HABITABLE_ROOMS',
                     'FLOOR_ENERGY_EFF',
                     'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF', 'ROOF_DESCRIPTION', 'ROOF_ENERGY_EFF',
                     'MAINHEATC_ENERGY_EFF', 'LIGHTING_DESCRIPTION', 'LIGHTING_ENERGY_EFF', 'CONSTRUCTION_AGE_BAND',
                     'TENURE']
    return model


def model_predict(model, feature_data):
    predictions = model.energy_model_predict(feature_data)
    return predictions


if __name__ == "__main__":
    model = model_training()

    data = pd.read_csv('./Training data.csv').drop(columns=['Index'])
    target = data['ENERGY_CONSUMPTION_CURRENT']
    features = data.drop(columns=['ENERGY_CONSUMPTION_CURRENT'])
    prediction = model_predict(model, features)


    # 计算R^2值
    r_squared = r2_score(target, prediction)
    print(f"R^2 Score: {r_squared}")

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.scatter(target, prediction, color='blue', label='Predicted vs Actual')
    plt.plot([target.min(), target.max()], [target.min(), target.max()], color='red', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Energy Consumption')
    plt.ylabel('Predicted Energy Consumption')
    plt.title('Fit of the Regional Linear Models')
    plt.legend()
    plt.show()
