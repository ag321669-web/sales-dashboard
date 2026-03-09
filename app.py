import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Title
st.title("📊 Sales Analytics & Forecasting Dashboard")

# Sidebar upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Default dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.DataFrame({
        "Month": [1,2,3,4,5],
        "Sales": [200,220,250,300,350]
    })

# Sales Summary
total_sales = data["Sales"].sum()
avg_sales = data["Sales"].mean()
max_sales = data["Sales"].max()
min_sales = data["Sales"].min()

st.subheader("Sales Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", int(total_sales))
col2.metric("Average Sales", int(avg_sales))
col3.metric("Highest Sales", int(max_sales))
col4.metric("Lowest Sales", int(min_sales))

# Edit dataset
st.subheader("Edit Dataset")

edited_data = st.data_editor(data, num_rows="dynamic")
data = edited_data

# Show dataset
st.subheader("Sales Dataset")
st.write(data)

# Download dataset
st.subheader("Download Sales Data")

csv = data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="sales_data.csv",
    mime="text/csv"
)

# Manual data entry
st.sidebar.header("Add New Data")

month_input = st.sidebar.number_input("Enter Month", min_value=1, max_value=12, step=1)
sales_input = st.sidebar.number_input("Enter Sales", min_value=0)

if st.sidebar.button("Add Data"):
    new_row = {"Month": month_input, "Sales": sales_input}
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    st.sidebar.success("Data Added Successfully!")

# Filter
st.sidebar.header("Filter Options")

max_month = st.sidebar.slider(
    "Select number of months to display",
    min_value=1,
    max_value=12,
    value=12
)

filtered_data = data[data["Month"] <= max_month]

# Filter metrics
total_sales = filtered_data["Sales"].sum()
avg_sales = filtered_data["Sales"].mean()
max_sales = filtered_data["Sales"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", total_sales)
col2.metric("Average Sales", round(avg_sales,2))
col3.metric("Max Sales", max_sales)

# Bar chart
st.subheader("Monthly Sales Bar Chart")

plt.figure()
plt.bar(filtered_data["Month"], filtered_data["Sales"])
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Comparison")
st.pyplot(plt)

# Line chart
st.subheader("Sales Trend")

plt.figure()
plt.plot(filtered_data["Month"], filtered_data["Sales"], marker="o")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
st.pyplot(plt)

# Heatmap
st.subheader("Correlation Heatmap")

plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

# Machine Learning Model
X = data[["Month"]]
y = data["Sales"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

r2 = r2_score(y, predictions)
mse = mean_squared_error(y, predictions)

st.subheader("Model Performance")
st.write("R2 Score:", r2)
st.write("Mean Squared Error:", mse)

# Future prediction
st.subheader("Predict Future Sales")

future_month = st.number_input("Enter future month", min_value=1, max_value=24)

if future_month:
    prediction = model.predict([[future_month]])
    st.success(f"Predicted Sales: {int(prediction[0])}")

    result = pd.DataFrame({
        "Month": [future_month],
        "Predicted_Sales": [int(prediction[0])]
    })

    st.download_button(
        label="Download Prediction Report",
        data=result.to_csv(index=False),
        file_name="sales_prediction.csv",
        mime="text/csv"
    )