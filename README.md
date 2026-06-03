# Student-Performance-Prediction-using-Machine-Learning-


Developed a ML model to predict students' math performance using demographic and academic factors. Performed data preprocessing, exploratory data analysis (EDA), feature scaling, and trained multiple regression models including Linear Regression, Random Forest, Gradient Boosting, and SVR. Evaluated model performance using R² Score,MAE,RMSE
#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#modeling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings
df=pd.read_csv('Studentdata.csv')
print(df)
df.head(5)
X=df.drop(columns=['math score'],axis=1)
X
X.head()
y=df['math score']
y
print("categories in 'gender' variable:   ", end=" ")
print(df['gender'].unique())
print("categories in 'race/ethnicity' variable:   ", end=" ")
print(df['race/ethnicity'].unique())
print("categories in 'parental level of education	' variable:   ", end=" ")
print(df['parental level of education'].unique())
print("categories in 'lunch' variable:   ", end=" ")
print(df['lunch'].unique())
print("categories in 'test preparation course	' variable:   ", end=" ")
print(df['test preparation course'].unique())

num_cols=x.select_dtypes(exclude="object").columns
cat_cols=x.select_dtypes(include="object").columns
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
num_trans=StandardScaler()
oh_tran=OneHotEncoder()
preprocessor=ColumnTransformer(
    [
        ("OneHotEncoder",oh_tran,cat_cols),
        ("StandardScaler",num_trans,num_cols),
    ]
)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)

# STUDENT PERFORMANCE - COMPLETE MACHINE LEARNING PROJECT
# STEP 1: IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
print("=" * 60)
print("  STUDENT PERFORMANCE ML PROJECT")
print("=" * 60)
# STEP 2: LOAD DATA 
df = pd.read_csv("Studentdata.csv")
 
print("\n STEP 2: DATA LOADING")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head()) 
# STEP 3: DATA CHECKS 
print("\n STEP 3: DATA CHECKS")
print("\n Data Types:")
print(df.dtypes)
 
print("\n Missing Values:")
print(df.isnull().sum())
 
print("\n Duplicate Rows:", df.duplicated().sum())
 
print("\n Statistical Summary:")
print(df.describe()) 
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA) 
print("\n STEP 4: EDA - Generating Plots...")
 
# 4.1 Score Distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Score Distributions', fontsize=16, fontweight='bold')
 
for ax, col, color in zip(axes,
                           ['math score', 'reading score', 'writing score'],
                           ['royalblue', 'tomato', 'seagreen']):
    ax.hist(df[col], bins=20, color=color, edgecolor='black', alpha=0.8)
    ax.set_title(col.title(), fontsize=13)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean: {df[col].mean():.1f}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
 
plt.tight_layout()
plt.savefig('01_score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: 01_score_distributions")
 
# 4.2 Gender vs Scores
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Average Scores by Gender', fontsize=16, fontweight='bold')
 
for ax, col, color in zip(axes,
                           ['math score', 'reading score', 'writing score'],
                           [['#4472C4', '#ED7D31'], ['#70AD47', '#FF0000'], ['#7030A0', '#FF69B4']]):
    df.groupby('gender')[col].mean().plot(kind='bar', ax=ax, color=color, edgecolor='black', rot=0)
    ax.set_title(col.title(), fontsize=13)
    ax.set_xlabel('Gender', fontsize=11)
    ax.set_ylabel('Average Score', fontsize=11)
    ax.legend(['Female', 'Male'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
 
plt.tight_layout()
plt.savefig('02_gender_vs_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_gender_vs_scores")
 
# 4.3 Lunch vs Scores
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Average Scores by Lunch Type', fontsize=16, fontweight='bold')
 
for ax, col in zip(axes, ['math score', 'reading score', 'writing score']):
    df.groupby('lunch')[col].mean().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', rot=0)
    ax.set_title(col.title(), fontsize=13)
    ax.set_xlabel('Lunch Type', fontsize=11)
    ax.set_ylabel('Average Score', fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
 
plt.tight_layout()
plt.savefig('03_lunch_vs_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_lunch_vs_scores")
 
# 4.4 Test Prep vs Scores
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Average Scores by Test Preparation Course', fontsize=16, fontweight='bold')
 
for ax, col in zip(axes, ['math score', 'reading score', 'writing score']):
    df.groupby('test preparation course')[col].mean().plot(
        kind='bar', ax=ax, color=['#F39C12', '#27AE60'], edgecolor='black', rot=0)
    ax.set_title(col.title(), fontsize=13)
    ax.set_xlabel('Test Prep Course', fontsize=11)
    ax.set_ylabel('Average Score', fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
 
plt.tight_layout()
plt.savefig('04_testprep_vs_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_testprep_vs_scores")
 
# 4.5 Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[['math score', 'reading score', 'writing score']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=1, square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of Scores', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_correlation_heatmap")
 
# 4.6 Race/Ethnicity vs Math Score
plt.figure(figsize=(10, 5))
df.groupby('race/ethnicity')['math score'].mean().sort_values().plot(
    kind='bar', color='steelblue', edgecolor='black')
plt.title('Average Math Score by Race/Ethnicity', fontsize=14, fontweight='bold')
plt.xlabel('Race/Ethnicity Group', fontsize=12)
plt.ylabel('Average Math Score', fontsize=12)
plt.xticks(rotation=0)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('06_ethnicity_math.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_ethnicity_math") 
# STEP 5: DATA PREPROCESSING 
print("\n📌 STEP 5: DATA PREPROCESSING")
 
df_encoded = df.copy()
le = LabelEncoder()
cat_cols = ['gender', 'race/ethnicity', 'parental level of education',
            'lunch', 'test preparation course']
 
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"Encoded: {col}")
 
print("\nEncoded DataFrame head:")
print(df_encoded.head())
 
# Features and Target
X = df_encoded.drop(columns=['math score', 'reading score', 'writing score'])
y = df_encoded['math score']   # Target: Math Score
 
print(f"\n  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
 
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 
print(f"\n  Train size: {X_train.shape[0]} samples")
print(f"  Test size : {X_test.shape[0]} samples")
 
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Feature Scaling Done (StandardScaler)") 
# STEP 6: MODEL TRAINING & EVALUATION 
print("\n STEP 6: MODEL TRAINING & RESULTS")
print("-" * 55)
print(f"{'Model':<30} {'R2 Score':>10} {'MAE':>8} {'RMSE':>8}")
print("-" * 55)
 
models = {
    "LR"  : LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "KNN"  : KNeighborsRegressor(),
    "DT"   : DecisionTreeRegressor(),
    "Rf"   : RandomForestRegressor(),
    "GBR"  : GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "SVR"  : SVR(),
}
 
results = {}
predictions = {}
 
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    predictions[name] = y_pred
    print(f"{name:<30} {r2:>10.4f} {mae:>8.2f} {rmse:>8.2f}")
 
print("-" * 55) 
# STEP 7: RESULTS VISUALIZATION 
print("\nSTEP 7: RESULT PLOTS")
 
# 7.1 R2 Score Bar Chart - All Models
model_names = list(results.keys())
r2_scores   = [results[m]['R2'] for m in model_names]
 
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, r2_scores,
               color=['#2196F3','#FF5722','#4CAF50','#9C27B0',
                      '#FF9800','#009688','#F44336','#795548','#607D8B'],
               edgecolor='black', width=0.6)
plt.title('R² Score Comparison - All Models', fontsize=15, fontweight='bold')
plt.xlabel('Model Name', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend(bars, model_names, title='Models', loc='upper right', fontsize=9)
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('07_r2_score_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_r2_score_comparison")
 
# 7.2 Linear Regression: y_pred vs y_test (Line Plot)
lr_pred = predictions['LR']
x_axis  = np.arange(len(y_test))
 
plt.figure(figsize=(14, 5))
plt.plot(x_axis, list(y_test), color='royalblue', linewidth=1.5,
         label='Actual (y_test)', linestyle='-')
plt.plot(x_axis, lr_pred, color='orangered', linewidth=1.5,
         label='Predicted (y_pred)', linestyle='--')
plt.title('Linear Regression: Actual vs Predicted Math Scores', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Math Score', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('08_LR_actual_vs_predicted_line.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 08_LR_actual_vs_predicted_line")
 
# 7.3 SNS Regression Plot: y_test vs y_pred for Math Score
plt.figure(figsize=(8, 6))
sns.regplot(x=list(y_test), y=lr_pred,
            scatter_kws={'color': 'steelblue', 'alpha': 0.6, 's': 50, 'edgecolors': 'black'},
            line_kws={'color': 'red', 'linewidth': 2},
            label='Regression Line')
plt.title('SNS Regression Plot: Actual vs Predicted (Math Score)', fontsize=13, fontweight='bold')
plt.xlabel('Actual Math Score (y_test)', fontsize=12)
plt.ylabel('Predicted Math Score (y_pred)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('09_sns_regplot_math_score.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 09_sns_regplot_math_score")
 
# 7.4 Difference: Actual - Predicted (Linear Regression)
diff = np.array(list(y_test)) - lr_pred
x_axis = np.arange(len(diff))
 
plt.figure(figsize=(14, 5))
plt.bar(x_axis, diff,
        color=['tomato' if d < 0 else 'seagreen' for d in diff],
        edgecolor='none', alpha=0.8, label='Difference (Actual - Predicted)')
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.title('Difference Between Actual and Predicted Math Scores (Linear Regression)',
          fontsize=13, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Difference (Actual - Predicted)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('10_difference_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 10_difference_actual_vs_predicted")
 
# 7.5 Scatter: Actual vs Predicted for ALL Models
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Actual vs Predicted Math Scores - All Models', fontsize=16, fontweight='bold')
axes = axes.flatten()
 
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0',
          '#FF9800','#009688','#F44336','#795548','#607D8B']
 
for i, (name, y_pred) in enumerate(predictions.items()):
    axes[i].scatter(list(y_test), y_pred, color=colors[i], alpha=0.6,
                    edgecolors='black', s=40, label='Predicted vs Actual')
    axes[i].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 color='black', linestyle='--', linewidth=1.5, label='Perfect Fit')
    axes[i].set_title(f'{name} (R²={results[name]["R2"]:.3f})', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Actual Score', fontsize=10)
    axes[i].set_ylabel('Predicted Score', fontsize=10)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, linestyle='--', alpha=0.5)
 
plt.tight_layout()
plt.savefig('11_all_models_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 11_all_models_scatter")
 
# 7.6 MAE & RMSE Comparison
mae_scores  = [results[m]['MAE']  for m in model_names]
rmse_scores = [results[m]['RMSE'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
 
fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, mae_scores,  width, label='MAE',  color='#42A5F5', edgecolor='black')
bars2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE', color='#EF5350', edgecolor='black')
ax.set_title('MAE & RMSE Comparison - All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Error Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
for b in bars1:
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.1, f'{b.get_height():.2f}',
            ha='center', va='bottom', fontsize=8)
for b in bars2:
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.1, f'{b.get_height():.2f}',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('12_mae_rmse_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(" saved: 12_mae_rmse_comparison") 
# STEP 8: BEST MODEL 
best_model = max(results, key=lambda m: results[m]['R2'])
print("\n" + "=" * 60)
print("  STEP 8: BEST MODEL")
print("=" * 60)
print(f"   Best Model  : {best_model}")
print(f"   R² Score    : {results[best_model]['R2']:.4f}")
print(f"   MAE         : {results[best_model]['MAE']:.2f}")
print(f"   RMSE        : {results[best_model]['RMSE']:.2f}")
print("=" * 60)
 
print("""
ALL STEPS COMPLETED SUCCESSFULLY!
Graphs saved:
  01_score_distributions.png
  02_gender_vs_scores.png
  03_lunch_vs_scores.png
  04_testprep_vs_scores.png
  05_correlation_heatmap.png
  06_ethnicity_math.png
  07_r2_score_comparison.png
  08_LR_actual_vs_predicted_line.png
  09_sns_regplot_math_score.png
  10_difference_actual_vs_predicted.png
  11_all_models_scatter.png
  12_mae_rmse_comparison.png
""")
