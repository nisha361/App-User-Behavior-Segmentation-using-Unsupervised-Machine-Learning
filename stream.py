import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="App User Segmentation", layout="wide")

st.title("📊 App User Behavior Segmentation Dashboard")
st.markdown("Unsupervised Machine Learning using K-Means Clustering")

# ----------------------
# Load Dataset
# ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\Nishadinesh\\Downloads\\app_user_behavior_dataset.csv")


df = load_data()

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.header("Dashboard Controls")
num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 6, 4)

# ----------------------
# Feature Selection
# ----------------------
features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'daily_active_minutes',
    'feature_clicks_per_session',
    'pages_viewed_per_session',
    'notifications_opened_per_week',
    'in_app_search_count',
    'content_downloads',
    'social_shares',
    'days_since_last_login',
    'churn_risk_score',
    'engagement_score'
]

X = df[features]

# ----------------------
# Scaling
# ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------
# K-Means Clustering
# ----------------------
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------
# KPI Section
# ----------------------
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", df.shape[0])
col2.metric("Clusters", num_clusters)
col3.metric("Avg Engagement", round(df['engagement_score'].mean(), 2))
col4.metric("Avg Churn Risk", round(df['churn_risk_score'].mean(), 2))

# ----------------------
# Cluster Distribution
# ----------------------
st.subheader("👥 User Distribution by Cluster")
fig1, ax1 = plt.subplots()
sns.countplot(x='cluster', data=df, ax=ax1)
st.pyplot(fig1)

# ----------------------
# Cluster Profiling Table
# ----------------------
st.subheader("📊 Cluster Behavioral Summary")
cluster_profile = df.groupby('cluster')[features].mean().round(2)
st.dataframe(cluster_profile)

# ----------------------
# PCA Visualization
# ----------------------
st.subheader("🧭 Cluster Visualization (PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x='pca1', y='pca2',
    hue='cluster',
    palette='Set2',
    data=df,
    ax=ax2
)
ax2.set_title("User Segments using PCA")
st.pyplot(fig2)

# ----------------------
# Cluster Explorer
# ----------------------
st.subheader("🔍 Explore Individual Clusters")
selected_cluster = st.selectbox("Select Cluster", sorted(df['cluster'].unique()))

st.write(df[df['cluster'] == selected_cluster].head(50))

# ----------------------
# Business Recommendations
# ----------------------
st.subheader("💡 Business Recommendations")

if selected_cluster == 0:
    st.success("High Engagement Users → Loyalty programs, premium plans")
elif selected_cluster == 1:
    st.info("Moderate Users → Personalized notifications & feature discovery")
elif selected_cluster == 2:
    st.warning("Low Engagement Users → Retention offers & reactivation campaigns")
else:
    st.error("Occasional Users → Onboarding improvements & reminders")
