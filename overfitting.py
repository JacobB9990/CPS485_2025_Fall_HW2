import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeClassifier 
from sklearn.metrics import zero_one_loss

df = pd.read_csv("./data/data.csv")

feature_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
X = df[feature_cols]
y = df["Outcome"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

N_train = X_train_scaled.shape[0] 
max_degree = 6
results = []
ALPHA = 1e-9

for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    num_params = X_train_poly.shape[1]

    if num_params > 4500:
        break

    model = RidgeClassifier(alpha=ALPHA, random_state=42)
    model.fit(X_train_poly, y_train)

    train_error = zero_one_loss(y_train, model.predict(X_train_poly))
    test_error = zero_one_loss(y_test, model.predict(X_test_poly))

    results.append(
        {
            "Degree": degree,
            "Num_Parameters": num_params,
            "Train_Error": train_error,
            "Test_Error": test_error,
        }
    )

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(
    results_df["Num_Parameters"],
    results_df["Train_Error"],
    marker="o",
    label="Train Error (Risk)",
    color="blue",
)
plt.plot(
    results_df["Num_Parameters"],
    results_df["Test_Error"],
    marker="o",
    label="Test Error (Generalization Risk)",
    color="red",
)

interpolation_threshold = N_train
plt.axvline(
    x=interpolation_threshold,
    color="black",
    linestyle="--",
    label=f"Interpolation Threshold ($P \\approx {N_train}$)",
)

plt.title("Double Descent Phenomenon via Polynomial Feature Complexity")
plt.xlabel("Model Complexity (Number of Parameters, P)")
plt.ylabel("Error Rate (Classification Loss)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0)
plt.savefig("./out/overfitting_double_decent")
plt.show()
