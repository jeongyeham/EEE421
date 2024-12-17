import joblib
import pandas as pd
from pymoo.core.problem import Problem
import math
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import plotly.express as px

wall_insulation_costs = [1000, 5000, 12000, 14000]
heating_system_costs = [2000, 5000, 6500, 7000]
glazing_costs = [1500, 2000, 2500, 4000]
pv_costs = [0, 800, 1000, 1200, 1600, 2400, 2800, 3600, 4000]

wall_insulation_lifetime = 30
heating_system_lifetime = 16
glazing_lifetime = 20
pv_lifetime = 10

Discount_rate = 0.05  # 折损率
Tariff = 0.25  # 电价


def region_model_load(model_path):
    return joblib.load(model_path)

def region_model_predict(feature_names, model):
    return model.predict(feature_names)

# current_energy_consumption prediction
def predict_current_energy_consumption():
    model = region_model_load('./model.joblib')
    feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                       'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                       'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']
    all_feature_data = pd.read_csv('./group_data.csv')
    X = pd.DataFrame(all_feature_data.iloc[0, all_feature_data.columns.get_indexer(feature_columns)]).T
    current_energy_consumption = region_model_predict(X, model).item()
    return current_energy_consumption  # 609


def calculate_epv(kWp, latitude, orientation, tilt, annual_solar_radiation):
    """
    Calculate the annual electricity production of a PV system.

    Parameters:
    - kWp: Installed peak power of the PV module (kWp)
    - latitude: Latitude of the location (degrees)
    - orientation: Orientation of the PV array (degrees from North, e.g., South is 180)
    - tilt: Tilt angle of the PV array (degrees from horizontal)
    - annual_solar_radiation: Annual solar radiation (kWh/m²)

    Returns:
    - Epv: Annual electricity production (kWh)
    """
    # Constants for calculation
    k1 = 26.3
    k2 = -38.5
    k3 = 14.8
    k4 = -16.5
    k5 = 27.3
    k6 = -11.9
    k7 = -1.06
    k8 = 0.0872
    k9 = -0.191

    # Convert orientation to radians
    orientation_rad = math.radians(orientation)

    # Calculate A, B, C for converting from horizontal to vertical or inclined solar flux
    A = (k1 * math.sin(tilt / 2) ** 2 + k2 * math.sin(tilt / 2) + k3 * math.sin(tilt / 2))
    B = (k4 * math.sin(tilt / 2) ** 2 + k5 * math.sin(tilt / 2) + k6 * math.sin(tilt / 2))
    C = (k7 * math.sin(tilt / 2) ** 2 + k8 * math.sin(tilt / 2) + k9 * math.sin(tilt / 2) + 1)

    # Calculate the factor for converting from horizontal to vertical or inclined solar flux
    R_horizontal_to_inclined = A * math.cos(latitude - math.radians(180)) + B * math.cos(
        2 * (latitude - math.radians(180))) + C

    # Calculate the annual electricity production
    Epv = 0.8 * kWp * annual_solar_radiation * R_horizontal_to_inclined

    return Epv





def calculate_annual_solar_radiation(region, orientation):
    """
    Calculate the annual solar radiation for a given region and orientation.

    Parameters:
    - region: The region number as per Table U6 in SAP 10.2
    - orientation: The orientation of the surface (N, NE, E, SE, S, SW, W, NW)

    Returns:
    - annual_solar_radiation: Annual solar radiation (kWh/m²)
    """
    # Solar radiation data for each month and region (from Table U3)
    solar_radiation_data = {
        1: [26, 54, 96, 150, 192, 200, 189, 157, 115, 66, 33, 21],  # UK average
        2: [30, 56, 98, 157, 195, 217, 203, 173, 127, 73, 39, 24],  # Thames
        3: [32, 59, 104, 170, 208, 231, 216, 182, 133, 77, 41, 25],  # South East England
        4: [35, 62, 109, 172, 209, 235, 217, 185, 138, 80, 44, 27],  # Southern England
        5: [36, 63, 110, 174, 210, 233, 204, 182, 136, 78, 44, 28],  # South West England
        6: [32, 59, 105, 167, 201, 226, 206, 175, 130, 74, 40, 25],  # Severn Wales / Severn England
        7: [28, 55, 97, 153, 191, 208, 194, 163, 121, 69, 35, 23],  # Midlands
        8: [24, 51, 95, 152, 191, 203, 186, 152, 115, 65, 31, 20],  # West Pennines Wales / West Pennines England
        9: [23, 51, 95, 157, 200, 203, 194, 156, 113, 62, 30, 19],  # North West England / South West Scotland
        10: [23, 50, 92, 151, 200, 196, 187, 153, 111, 61, 30, 18],  # Borders Scotland / Borders England
        11: [25, 51, 95, 152, 196, 198, 190, 156, 115, 64, 32, 20],  # North East England
        12: [26, 54, 96, 150, 192, 200, 189, 157, 115, 66, 33, 21],  # East Pennines
        13: [30, 58, 101, 165, 203, 220, 206, 173, 128, 74, 39, 24],  # East Anglia
        14: [29, 57, 104, 164, 205, 220, 199, 167, 120, 68, 35, 22],  # Wales
        15: [19, 46, 88, 148, 196, 193, 185, 150, 101, 55, 25, 15],  # West Scotland
        16: [21, 46, 89, 146, 198, 191, 183, 150, 106, 57, 27, 15],  # East Scotland
        17: [19, 45, 89, 143, 194, 188, 177, 144, 101, 54, 25, 14],  # North East Scotland
        18: [17, 43, 85, 145, 189, 185, 170, 139, 98, 51, 22, 12],  # Highland
        19: [16, 41, 87, 155, 205, 206, 185, 148, 101, 51, 21, 11],  # Western Isles
        20: [14, 39, 84, 143, 205, 201, 178, 145, 100, 50, 19, 9],  # Orkney
        21: [12, 34, 79, 135, 196, 190, 168, 144, 90, 46, 16, 7],  # Shetland
        22: [24, 52, 96, 155, 201, 198, 183, 150, 107, 61, 30, 18]  # Northern Ireland
    }

    # Calculate the annual solar radiation for the given region and orientation
    annual_solar_radiation = sum(solar_radiation_data[region]) / 12  # Average monthly solar radiation

    # Adjust for orientation (this is a simplified approach and may need to be adjusted based on the specific orientation)
    orientation_factors = {
        'N': 0.9,
        'NE': 0.95,
        'E': 1.0,
        'SE': 1.05,
        'S': 1.1,
        'SW': 1.05,
        'W': 1.0,
        'NW': 0.95
    }

    annual_solar_radiation *= orientation_factors[orientation]

    return annual_solar_radiation


class EnergyCostOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=4,  # 决策变量数量
                         n_obj=2,  # 目标函数数量 (成本和能耗)
                         n_constr=0,  # 约束数量
                         xl=np.array([3, 1, 0, 0]),  # 决策变量下界
                         xu=np.array([3, 3, 3, 7]))  # 决策变量上界

        self.model = region_model_load('./model.joblib')
        self.feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                                'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                                'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']
        self.all_feature_data = pd.read_csv('./group_data.csv')
        self.X = pd.DataFrame(self.all_feature_data.iloc[0, self.all_feature_data.columns.get_indexer(self.feature_columns)]).T
        self.current_energy_consumption = predict_current_energy_consumption()

    def _evaluate(self, X, out, *args):
        # 初始化目标函数值
        npvs = []
        energies = []
        constrains = []


        # 遍历每个解 X[i]
        for x in X:
            # 决策变量
            wall_level = int(x[0])
            heat_level = int(x[1])
            glazing_level = int(x[2])
            pv_count = int(x[3])

            # 根据当前解的决策变量更新 self.X 中的特征
            updated_X = self.X.copy()  # 深拷贝，以避免修改原始数据
            updated_X['WALLS_ENERGY_EFF'] = wall_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_wall = ((energy_predictions - self.current_energy_consumption) * Tariff) * self.all_feature_data['TOTAL_FLOOR_AREA']

            updated_X = self.X.copy()
            updated_X['MAINHEAT_ENERGY_EFF'] = heat_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_heat = ((energy_predictions - self.current_energy_consumption) * Tariff) * self.all_feature_data['TOTAL_FLOOR_AREA']

            updated_X = self.X.copy()
            updated_X['GLAZED_TYPE'] = glazing_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_glazing = ((energy_predictions-self.current_energy_consumption) * Tariff) * self.all_feature_data['TOTAL_FLOOR_AREA']

            # 能效预测，传递更新后的特征
            updated_X = self.X.copy()
            updated_X['WALLS_ENERGY_EFF'] = wall_level
            updated_X['MAINHEAT_ENERGY_EFF'] = heat_level
            updated_X['GLAZED_TYPE'] = glazing_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            # energy_saving_transfer_benifit_per_year =( energy_predictions *  Tariff ) * self.all_feature_data['TOTAL_FLOOR_AREA'] + calculate_epv() * Tariff # 每省下来的能源折算成钱

            # 初始设备成本
            wall_cost = wall_insulation_costs[wall_level]
            heat_cost = heating_system_costs[heat_level]
            glazing_cost = glazing_costs[glazing_level]
            pv_cost = pv_costs[pv_count]

            toal_inital_cost = wall_cost + heat_cost + glazing_cost + pv_cost

            # 设备折现成本：分别计算每个设备的折现成本
            discounted_wall_benifit = 0
            for year in range(1, wall_insulation_lifetime + 1):
                discounted_wall_benifit += energy_saving_transfer_benifit_per_year_wall / ((1 + Discount_rate) ** year)

            discounted_heat_benifit = 0
            for year in range(1, heating_system_lifetime + 1):
                discounted_heat_benifit += energy_saving_transfer_benifit_per_year_heat / ((1 + Discount_rate) ** year)

            discounted_glazing_benifit = 0
            for year in range(1, glazing_lifetime + 1):
                discounted_glazing_benifit += energy_saving_transfer_benifit_per_year_glazing / ((1 + Discount_rate) ** year)

            discounted_pv_benifit = 0
            kWp = 0.32 * pv_count  # 1 kWp
            latitude = 51.5  # Example latitude for Severn Wales / Severn England
            orientation = 135  # South
            tilt = 45 # 45 degrees from horizontal
            annual_solar_radiation = calculate_annual_solar_radiation(14, 'SW')
            for year in range(1, pv_lifetime + 1):
                discounted_pv_benifit +=  calculate_epv(kWp, latitude, orientation, tilt,annual_solar_radiation) * Tariff / ((1 + Discount_rate) ** year)

            # 总设备折现成本：将所有设备的折现成本相加
            total_discounted_device_benifit = (discounted_wall_benifit +
                                            discounted_heat_benifit +
                                            discounted_glazing_benifit +
                                            discounted_pv_benifit)

            # 总成本 (设备折现成本 + 折现后的电费)
            total_npv = total_discounted_device_benifit - toal_inital_cost

            # 总能源节省
            total_energy_saving = ((energy_predictions - self.current_energy_consumption) * self.all_feature_data['TOTAL_FLOOR_AREA'] + calculate_epv(kWp, latitude, orientation, tilt,annual_solar_radiation)) / self.all_feature_data['TOTAL_FLOOR_AREA']  # kwh/m2

            # 保存结果
            npvs.append(-total_npv)
            energies.append(-total_energy_saving)

            #constrains.append(total_energy_saving)

            # 将结果存储到 out 中，供 NSGA-II 使用
        out["F"] = np.column_stack([npvs, energies])
        #out["G"] = np.column_stack([constrains])


##############################################NSGA####################################################################


if __name__ == "__main__":

    # 创建优化问题
    problem = EnergyCostOptimization()

    # 定义 NSGA-II 算法
    algorithm = NSGA2(pop_size=100)

    # 运行优化
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),  # 运行200代
                   verbose=False)
    # 打印结果
    print("Pareto-optimal solutions:")
    for sol in res.F:
        print(f"NPV: {-sol[0]:.2f}, Energy: {-sol[1]:.2f}")

    # 打印最优解的决策变量分配
    print("Pareto-optimal solutions (Decision Variables):")
    for i, sol in enumerate(res.X):
        wall_level = int(sol[0])  # 墙体保温等级
        heat_level = int(sol[1])  # 供热系统等级
        glazing_level = int(sol[2])  # 玻璃窗等级
        pv_count = int(sol[3])  # 光伏数量

    print(f"Solution {i + 1}: Wall Insulation Level: {wall_level}, Heating System Level: {heat_level}, "
          f"Glazing Level: {glazing_level}, PV Count: {pv_count}")

    # 构造 DataFrame 来存储每个解的信息
    solutions = []
    for i, (decision_vars, objectives) in enumerate(zip(res.X, res.F)):
        wall_level = int(decision_vars[0])  # 墙体保温等级
        heat_level = int(decision_vars[1])  # 供热系统等级
        glazing_level = int(decision_vars[2])  # 玻璃窗等级
        pv_count = int(decision_vars[3])  # 光伏数量

        cost = -objectives[0]
        energy = -objectives[1]  # 转为正值表示节能

        # 将每个解的决策变量和目标值存储为字典
        solutions.append({
            "Solution #": i + 1,
            "Wall Insulation Level": wall_level,
            "Heating System Level": heat_level,
            "Glazing Level": glazing_level,
            "PV Count": pv_count,
            "NPV": round(cost, 2),
            "Energy Saving": round(energy, 2)
        })

    # 将结果转换为 DataFrame
    df_solutions = pd.DataFrame(solutions)

    # 使用 Plotly 绘制交互式散点图
    fig = px.scatter(
        df_solutions,
        x="NPV",
        y="Energy Saving",
        hover_data=["Solution #", "Wall Insulation Level", "Heating System Level", "Glazing Level", "PV Count"],
        title="Pareto Front - NPV vs Energy Saving",
        labels={"NPV": "Total NPV", "Energy Saving": "Energy Saving (kWh/mm2)"},
        template="plotly_dark"
    )

    # 添加标签
    fig.update_traces(marker=dict(size=10, color='red'), selector=dict(mode='markers'))

    # 显示图像
    fig.show()

    fig.write_html("./pareto_front.html")
    print("Interactive plot saved as 'pareto_front.html'.")



#############################################################################################################################################
