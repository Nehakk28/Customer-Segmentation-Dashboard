import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

segments = {
0:("Budget Customers","Low income and low spending behavior."),
1:("Premium Customers","High income and high spending customers."),
2:("Average Customers","Moderate income and moderate spending."),
3:("High Value Customers","High income customers with strategic spending."),
4:("Low Engagement Customers","High income but low spending activity.")
}

def get_cluster_graph():

    fig = px.scatter(
        data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        title="Customer Segmentation"
    )

    return fig.to_html(full_html=False)

def predict_segment(income, score):

    cluster = kmeans.predict([[income,score]])[0]

    name, explanation = segments[cluster]

    return name, explanation

def get_statistics():

    stats = data.groupby("Cluster").size()

    return stats