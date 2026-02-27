import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report 

pd.set_option("display.max_columns", None)
sns.set_theme(style="darkgrid")

RANDOM_STATE = 42

df = pd.read_csv("sonar_data.csv", header=None)
df.head()

df.shape
df.info()
df.isnull().sum()
df.columns

for col in df.columns:
    print(df[col].value_counts())
    
    print(df[60].value_counts())
print("-"*40)
print(df[60].value_counts(normalize=True)*100)
df[60] = df[60].map({"R": 0, "M": 1})

print(df[60].value_counts())
print("-"*40)
print(df[60].value_counts(normalize=True)*100)

df.head()

duplicate_mask = df.duplicated()
num_duplicates = duplicate_mask.sum()

print("Number of dulicate rows:", num_duplicates)
print("Duplicated row:\n", df[duplicate_mask]) 

n_rows = len(df)
nunique = df.nunique()

constant_cols = nunique[nunique == 1].index.tolist()
print("Constant columns:", constant_cols)

# Quasi constant: top value more than 95 percent
quasi_constant_cols = []

for col in df.columns:
    top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
    if top_freq > 0.95 and col not in constant_cols:
        quasi_constant_cols.append(col)

print("Quasi constant columns (top value more than 95 percent):", quasi_constant_cols)
df.groupby(60).mean()

df.describe()

fig, axes = plt.subplots(13, 5, figsize=(15,18))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col, fontsize=8)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(13, 5, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col, fontsize=8)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.show()

X = df.drop(columns=[60])
y = df[60]

X.head()

y.head()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


print("Dataset Shape:", X.shape)
print("Training Dataset Shape:", X_train.shape)
print("Test Dataset Shape:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC()

model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Training Data Accuracy: {round(train_acc*100, 2)}%")

y_test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Data Accuracy: {round(test_acc*100, 2)}%")

print("Training Data - Classification Report")
print(classification_report(y_train, y_train_pred))

print("Test Data - Classification Report")
print(classification_report(y_test, y_test_pred))


def predict_object(input_features):
    # scale features
    scaled_features = scaler.transform([input_features])
    # get prediction from teh model
    prediction = model.predict(scaled_features)
    print("Model prediction", prediction)
    if prediction[0] == 1:
        print("The object is identified as Mine 💣")
    else:
        print("The object is identified as Rock 🪨")

X_test.head()

y_test.head()

test_1 = X_test.loc[103].tolist()
print(test_1)

predict_object(test_1)

test_2 = X_test.loc[90].tolist()
print(test_2)

predict_object(test_2)