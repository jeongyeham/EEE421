import keras
import numpy as np
import pandas as pd
from joblib import dump, load
from keras import Model
from keras.src.layers import Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam, Adamax
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from statsmodels.api import qqplot

class EnergyModel(Model):
    def __init__(self):
        super().__init__()

        self.preprocessor = None
        self.model_path = "./build/model.keras"
        self.training_file_path = "./Training data.csv"
        self.preprocessor_path = "./build/setting/preprocessor.pkl"
        self.target_scaler_path = "./build/setting/target_scaler.pkl"

        self.base_year = 1950
        self.base_postcode = 53

        self.feature_data = None
        self.target_data = None

        self.dense_1 = Dense(512, activation='relu')
        self.dropout_1 = Dropout(0.3)
        self.dense_2 = Dense(512, activation='relu')
        self.dropout_2 = Dropout(0.3)
        self.dense_3 = Dense(256, activation='relu')
        self.dropout_3 = Dropout(0.3)
        self.dense_4 = Dense(1, activation='linear')

        self.target_scaler = StandardScaler()

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_3(x)
        x = self.dropout_3(x, training=training)
        return self.dense_4(x)

    def build(self, input_shape):
        super().build(input_shape)

    def preprocess_training_data(self):
        data = pd.read_csv(self.training_file_path).drop(columns=['Index'])
        target_data = data['ENERGY_CONSUMPTION_CURRENT']
        feature_data = data.drop(columns=['ENERGY_CONSUMPTION_CURRENT'])

        feature_data['POSTCODE_DIFF'] = feature_data['POSTCODE'] - self.base_postcode
        feature_data = feature_data.drop(columns=['POSTCODE'])

        # 处理时间年份特征
        feature_data['YEAR_DIFF'] = feature_data['CONSTRUCTION_AGE_BAND'] - self.base_year
        feature_data = feature_data.drop(columns=['CONSTRUCTION_AGE_BAND'])

        # 定义数值和分类特征
        numeric_features = ['TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS',
                            'NUMBER_HEATED_ROOMS', 'YEAR_DIFF', 'POSTCODE_DIFF']
        
        categorical_features = ['PROPERTY_TYPE', 'BUILT_FORM', 'TRANSACTION_TYPE', 'ENERGY_TARIFF', 'FLOOR_LEVEL',
                                'GLAZED_TYPE', 'GLAZED_AREA', 'HOT_WATER_ENERGY_EFF', 'FLOOR_DESCRIPTION',
                                'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF', 'ROOF_DESCRIPTION',
                                'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF',
                                'LIGHTING_DESCRIPTION', 'LIGHTING_ENERGY_EFF', 'TENURE']
        
        energy_efficiency_features = ['CURRENT_ENERGY_EFFICIENCY', 'LOW_ENERGY_LIGHTING']
        # 定义预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('energy_eff', MinMaxScaler(), energy_efficiency_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        feature_data_scaled = self.preprocessor.fit_transform(feature_data)
        print(feature_data_scaled.shape)

        target_data_scaled = self.target_scaler.fit_transform(target_data.values.reshape(-1, 1))

        # 保存标准化器和预处理器
        dump(self.preprocessor, self.preprocessor_path)
        dump(self.target_scaler, self.target_scaler_path)

        return feature_data_scaled, target_data_scaled

    def energy_model_compile(self):
        super().compile(optimizer=Adamax(learning_rate=0.001), loss='huber', metrics=['mae', keras.metrics.R2Score()])

    def energy_model_train(self, feature_train_data, target_train_data):
        early_stopping = keras.callbacks.EarlyStopping(monitor='mae', patience=20, restore_best_weights=True)
        history = self.fit(feature_train_data, target_train_data, batch_size=32, epochs=400, validation_data=,
                           callbacks=[early_stopping])

        self.energy_model_save()
        return history

    def energy_model_save(self, overwrite=True):
        super().save(self.model_path, overwrite=overwrite)

    def energy_model_load(self):
        return keras.models.load_model(self.model_path)

    def energy_model_preprocess_new_data(self, new_data):
        new_data['POSTCODE_DIFF'] = new_data['POSTCODE'] - self.base_postcode
        new_data = new_data.drop(columns=['POSTCODE'])

        new_data['YEAR_DIFF'] = new_data['CONSTRUCTION_AGE_BAND'] - self.base_year
        new_data = new_data.drop(columns=['CONSTRUCTION_AGE_BAND'])

        self.preprocessor = load(self.preprocessor_path)
        new_data_scaled = self.preprocessor.transform(new_data)
        return new_data_scaled

    def energy_model_postprocess_predictions(self, predictions_scaled):
        self.target_scaler = load(self.target_scaler_path)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions

    def energy_model_predict(self, new_data):
        new_data_scaled = self.energy_model_preprocess_new_data(new_data)
        predictions_scaled = self.predict(new_data_scaled)
        predictions = self.energy_model_postprocess_predictions(predictions_scaled)
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


def model_predict_xlsx():
    data = pd.read_excel('./published_group_data_2024.xlsx').drop(columns=['Group_number'])
    data_feature = data.drop(columns=['ENERGY_CONSUMPTION_CURRENT']).iloc[:4]
    model = EnergyModel()
    feature_train_data, target_train_data = model.preprocess_training_data()
    model.energy_model_compile()
    model.energy_model_train(feature_train_data, target_train_data)
    predicted_data = model.energy_model_predict(data_feature)
    print(predicted_data)


def sensitivity_analyse(model, feature_data):
    feature_names = ['POSTCODE', 'CURRENT_ENERGY_EFFICIENCY', 'PROPERTY_TYPE', 'BUILT_FORM', 'TRANSACTION_TYPE',
                     'TOTAL_FLOOR_AREA', 'ENERGY_TARIFF', 'FLOOR_LEVEL', 'GLAZED_TYPE', 'GLAZED_AREA',
                     'NUMBER_HABITABLE_ROOMS',
                     'FLOOR_ENERGY_EFF',
                     'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF', 'ROOF_DESCRIPTION', 'ROOF_ENERGY_EFF',
                     'MAINHEATC_ENERGY_EFF', 'LIGHTING_DESCRIPTION', 'LIGHTING_ENERGY_EFF',
                     'TENURE']

    sensitivity_results = model.sensitivity_analysis(feature_data, feature_names)
    print(sensitivity_results)


if __name__ == "__main__":
    model_predict_xlsx()
    # 请确保提供正确的新数据格式
    # model = EnergyModel()
    # new_data = pd.read_excel('./published_group_data_2024.xlsx')
    # new_data = new_data.iloc[2, 17:18]  # 假设这是正确的切片
    # result = model.predict(new_data)
    # print(result)
