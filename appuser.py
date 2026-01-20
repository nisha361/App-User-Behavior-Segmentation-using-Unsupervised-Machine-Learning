import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


# Load GUVI App User Behavior dataset
df = pd.read_csv("C:\\Users\\Nishadinesh\\Downloads\\app_user_behavior_dataset.csv")


# Preview data
df.head()
df.info()
df.describe()
# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


# Drop duplicates
df.drop_duplicates(inplace=True)
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []


for k in range(2, 10):
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
inertia.append(kmeans.inertia_)


plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
cluster_summary = df.groupby('cluster')[features].mean()
cluster_summary
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]


plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
plt.title('User Clusters Visualization using PCA')
plt.show()
