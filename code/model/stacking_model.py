from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_stacking_model(X, y):
    """
    训练融合模型。
    
    参数:
    - X: 特征数据
    - y: 标签数据
    
    返回:
    - 训练好的融合模型
    """
    # 定义基础模型
    base_models = [
        ('LR', LogisticRegression(max_iter=1000)),
        ('DT', DecisionTreeClassifier()),
        ('SVM', SVC(probability=True)),
        ('ERT', ExtraTreesClassifier()),
        ('XGB', xgb.XGBClassifier(eval_metric='logloss'))
    ]
    
    # 定义元模型
    meta_model = RandomForestClassifier()
    
    # 创建融合模型
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3
    )
    
    # 训练融合模型
    stacking_model.fit(X, y)
    
    return stacking_model

def predict_probabilities(model, X):
    """
    使用模型对特征数据 X 进行概率预测。
    """
    return model.predict_proba(X)  # 返回概率

def evaluate_stacking_model(model, X_test, y_test):
    """
    评估融合模型的性能。
    
    参数:
    - model: 融合模型
    - X_test: 测试集特征数据
    - y_test: 测试集标签数据
    
    返回:
    - 准确率
    """
    # 使用模型进行预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    probabilities = predict_probabilities(model, X_test)
    print(f"Stacking Model Accuracy: {accuracy}")

    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability_0': probabilities[:, 0],
    })

    print(results_df)
    
    
    return accuracy