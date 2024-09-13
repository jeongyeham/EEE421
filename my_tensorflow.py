import joblib
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import statsmodels.api as sm
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BuildingEnergyAnalyzerDynamic:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._read_data()

    def _read_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def plot_each_column_as_scatter(self):
        if self.data is not None:
            for column in self.data.columns:
                fig = go.Figure(data=[go.Scatter(
                    x=self.data.index,
                    y=self.data[column],
                    mode='markers',
                    name=column
                )])
                fig.update_layout(
                    title=f'Scatter Plot of {column}',
                    xaxis_title='Row Number',
                    yaxis_title=column
                )
                fig.show()
        else:
            print("Data is not loaded. Cannot plot.")


class BuildingEnergyAnalyzerStatic:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._read_data()

    def _read_data(self):
        data = pd.read_csv(self.file_path, index_col=0)
        return data

    def plot_columns(self):
        if self.data is not None:
            for column in self.data.columns:
                plt.figure()  # 创建一个新的图形
                plt.scatter(range(self.data.shape[0]), self.data[column])
                plt.title(f'Scatter Plot of {column}')
                plt.xlabel('Row Number')
                plt.ylabel(column)
                plt.grid(True)
                plt.show()
        else:
            print("Data is not loaded. Cannot plot.")


class EnergyModel:
    def __init__(self, file_path='./Training data.csv', model_path='./setting/model.keras',
                 scaler_path='./setting/scaler.pkl'):
        self.data = pd.read_csv(file_path).drop(columns=['Index'])
        self.target = self.data['ENERGY_CONSUMPTION_CURRENT']
        self.features = self.data.drop(columns=['ENERGY_CONSUMPTION_CURRENT'])
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = model_path
        self.scaler_path = scaler_path

    def preprocess_data(self):
        features_scaled = self.scaler.fit_transform(self.features)
        target_scaled = self.scaler.fit_transform(self.target.values.reshape(-1, 1))
        return train_test_split(features_scaled, target_scaled, test_size=0.2)

    def build_model(self):
        self.model = Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    def train_model(self, feature_data, target_data):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        result = self.model.fit(feature_data, target_data, batch_size=32, epochs=1000, validation_split=0.2,
                                callbacks=[early_stopping])
        self.save_model()
        self.save_scaler()

        return result

    def save_model(self):
        self.model.save(self.model_path)

    def save_scaler(self):
        joblib.dump(self.scaler, self.scaler_path)

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_path)

    def load_model(self):
        self.load_model()

    def evaluate_model(self, feature_data, target_data):
        test_loss, test_mae = self.model.evaluate(feature_data, target_data)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    def predict(self, new_data):
        self.load_model()
        self.load_scaler()
        new_data_scaled = self.scaler.transform(new_data)
        predictions_scaled = self.model.predict(new_data_scaled)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        return predictions
    
    def sensitivity_analysis(self, feature_data, feature_name):
        baseline = self.model.predict(feature_data)
        results = {}
        for i, feature in enumerate(feature_name):
            original_value = feature_data[:, i]
            feature_data[:, i] = original_value * 1.1
            new_prediction = self.model.predict(feature_data)
            change = np.mean(new_prediction - baseline)
            results[feature] = change
            feature_data[:, i] = original_value
        return results


def data_analyse():
    analyzer = BuildingEnergyAnalyzerDynamic('./Training data.csv')
    # analyzer.plot_each_column_as_scatter()
    data = analyzer.data["TOTAL_FLOOR_AREA"]
    sm.qqplot(data, line='45')
    plt.show()

    # analyzer = BuildingEnergyAnalyzerStatic('./Training data.csv')
    # analyzer.plot_columns()


def model_trainning():
    model = EnergyModel()
    X_train, X_test, y_train, y_test = model.preprocess_data()
    model.build_model()
    history = model.train_model(X_train, y_train)
    model.evaluate_model(X_test, y_test)
    feature_names = ['POSTCODE', 'CURRENT_ENERGY_EFFICIENCY', 'PROPERTY_TYPE', 'BUILT_FORM', 'TRANSACTION_TYPE',
                     'TOTAL_FLOOR_AREA', 'ENERGY_TARIFF', 'FLOOR_LEVEL', 'GLAZED_TYPE', 'GLAZED_AREA',
                     'NUMBER_HABITABLE_ROOMS',
                     'FLOOR_ENERGY_EFF',
                     'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF', 'ROOF_DESCRIPTION', 'ROOF_ENERGY_EFF',
                     'MAINHEATC_ENERGY_EFF', 'LIGHTING_DESCRIPTION', 'LIGHTING_ENERGY_EFF', 'CONSTRUCTION_AGE_BAND',
                     'TENURE']
    


    sensitivity_results = model.sensitivity_analysis(X_test, feature_names)
    print(sensitivity_results)

if __name__ == "__main__":
    model_trainning()