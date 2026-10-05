import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/hextr/Desktop/data science/hours_studied_vs_grades.csv")

dftest = pd.read_csv("C:/Users/hextr/Desktop/data science/testdatahours.csv")

def findb(y,m,x):
    a = m*x
    b = y - a
    return b

def iter(x,y,xmean,ymean):
    
    productx = 0
    producty = 0

    prod = 0
    
    denom = 0
    

    for i,z in zip(x,y):
            productx = i - xmean
            producty = z - ymean
    
            prod += producty * productx
    
            denom += productx**2

    return prod/denom


def linearregression(x,y):
    xmean = np.mean(x)
    ymean = np.mean(y)

    m = iter(x,y,xmean,ymean)

    b = findb(ymean,m,xmean)

    return m, b


m,b = linearregression(df["Hours Studied"], df["Exam Grade (%)"])

# eval metrics

estimated_grade = (m * dftest["Hours Studied"]) + b
print(estimated_grade)

train_predictions = (m * df["Hours Studied"]) + b

actual_y = df["Exam Grade (%)"]
mse_total = np.sum((actual_y - train_predictions) ** 2)
mse = mse_total / len(actual_y)
rmse = np.sqrt(mse)

ss_tot = np.sum((actual_y - np.mean(actual_y)) ** 2)
r_squared = 1 - (mse_total / ss_tot)

print(f"\n--- Training Evaluation Metrics ---")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r_squared:.4f}")

# graphs

plt.scatter(df["Hours Studied"], df["Exam Grade (%)"], color="blue", label="Actual Data", zorder=3)


x_line = np.linspace(df["Hours Studied"].min(), df["Hours Studied"].max(), 100)
y_line = (m * x_line) + b


plt.plot(x_line, y_line, color="red", label=f'Regression line: y = {m:.1f}x + {b:.1f}', zorder=2)

plt.xlabel('Hours Studied')
plt.ylabel('Exam Grade (%)')
plt.title('Ordinary Least Squares (OLS) Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

dftest["Predicted Grade"] = (m * dftest["Hours Studied"]) + b

plt.figure(figsize=(8, 6))
plt.plot(x_line, y_line, color="red", linestyle="--", label="Trained Model Line", zorder=2)
plt.scatter(dftest["Hours Studied"], dftest["Predicted Grade"], color="green", s=100, label="Test Data Predictions", zorder=3)

plt.xlabel('Hours Studied')
plt.ylabel('Predicted Exam Grade (%)')
plt.title('Model Predictions on Test Dataset')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()