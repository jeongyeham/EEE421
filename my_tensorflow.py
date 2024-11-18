import pandas as pd
import numpy as np
import keras
from keras import Model
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from pygments.lexers import go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from matplotlib import pyplot as plt
from statsmodels.api import qqplot

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
                plt.figure()
                plt.scatter(range(self.data.shape[0]), self.data[column])
                plt.title(f'Scatter Plot of {column}')
                plt.xlabel('Row Number')
                plt.ylabel(column)
                plt.grid(True)
                plt.show()
        else:
            print("Data is not loaded. Cannot plot.")

class EnergyModel(Model):
    def __init__(self):
        super().__init__()
        self.features = None
        self.target = None
        self.dense_1 = Dense(25, activation='relu')
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(512, activation='relu')
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(128, activation='relu')
        self.dropout_3 = Dropout(0.2)
        self.dense_4 = Dense(1, activation='linear')
        self.scaler = StandardScaler()

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
        features_scaled = self.scaler.fit_transform(self.features)
        target_scaled = self.scaler.fit_transform(self.target.values.reshape(-1, 1))
        return train_test_split(features_scaled, target_scaled, test_size=0.1)

    def energy_model_compile(self):
        super().compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    def energy_model_train(self, feature_data, target_data):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        history = super().fit(feature_data, target_data, batch_size=64, epochs=1000, validation_split=0.2,
                                               callbacks=[early_stopping])
        return history

    def energy_model_save(self, model_path='./model.keras', overwrite=True):
        super().save(model_path, overwrite=overwrite)

    def energy_model_scaler_save(self, scaler_path='./scaler.pkl'):
        dump(self.scaler, scaler_path)

    def energy_model_scaler_load(self, scaler_path='./scaler.pkl'):
        self.scaler = load(scaler_path)

    def energy_model_load(self, model_path='./model.keras'):
        return keras.models.load_model(model_path)

    def energy_model_evaluate(self, feature_data, target_data):
        test_loss, test_mae = super().evaluate(feature_data, target_data)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    def energy_model_predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        predictions_scaled = super().predict(new_data_scaled)
        predictions = self.scaler.inverse_transform(predictions_scaled)
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

def data_analyse():
    analyzer = BuildingEnergyAnalyzerDynamic('./Training data.csv')
    data = analyzer.data["TOTAL_FLOOR_AREA"]
    qqplot(data, line='45')
    plt.show()

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

    #sensitivity_results = model.sensitivity_analysis(X_test, feature_names)
    #print(sensitivity_results)

if __name__ == "__main__":
    model_training()
    # 请确保提供正确的新数据格式
    # model = EnergyModel()
    # new_data = pd.read_excel('./published_group_data_2024.xlsx')
    # new_data = new_data.iloc[2, 17:18]  # 假设这是正确的切片
    # result = model.predict(new_data)
    # print(result)