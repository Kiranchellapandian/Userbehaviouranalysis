import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
file_path = "updated_sample_dataset_with_features.csv"
data = pd.read_csv(file_path)

# Encode categorical variables
categorical_columns = [
    "deviceType", "browserName", "osName", "screenResolution", "networkType",
    "locale", "userTier", "interestTags", "ageRange", "gender", "funnelStage",
    "abTestGroup", "community"
]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
numerical_columns = [
    "likeCount", "dislikeCount", "shareCount", "commentCount", "bookmarkCount",
    "ratingValue", "pageViews", "clickEvents", "scrollDepth", "timeOnPage",
    "sessionDuration", "sessionCount", "sentimentScore", "cartItemsCount",
    "followerCount", "followingCount", "groupMemberships", "mentionCount",
    "like_per_comment", "share_per_view", "interaction_intensity", "engagement_score",
    "activity_ratio"
]
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
data_pca = pca.fit_transform(data[numerical_columns])

# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Apply KMeans with the optimal number of clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(data_pca)

# Map cluster numbers to meaningful names
cluster_mapping = {
    0: "Sports",
    1: "News",
    2: "Technology",
    3: "Gaming",
    4: "Creators"
}
data['cluster_name'] = data['cluster'].map(cluster_mapping)

# Personalization Strategies Based on Cluster Names
def recommend_content(cluster_name):
    recommendations = {
        "Sports": "Challenges and interactive activities",
        "News": "Community events and discussions",
        "Technology": "Exclusive videos and blogs",
        "Gaming": "More articles and tutorials",
        "Creators": "User-generated content and forums"
    }
    return recommendations.get(cluster_name, "General recommendations")

data['personalization_strategy'] = data['cluster_name'].apply(recommend_content)

# Save the labeled dataset to a file
output_file = "improved_labeled_dataset.csv"
data.to_csv(output_file, index=False)

# Save the model and encoders
joblib.dump(kmeans, "improved_kmeans_model.pkl")
joblib.dump(pca, "pca_model.pkl")
joblib.dump(label_encoders, "categorical_encoders.pkl")

# Display Sample Labeled Output with Recommendations
sample_output = data[["likeCount", "shareCount", "interaction_intensity", "cluster_name", "community", "personalization_strategy"]].head(10)
print("\nSample Labeled User Data with Clustering and Personalization Strategies:")
print(sample_output.to_string(index=False))

# Save the full labeled dataset to a file
output_file = "labeled_output_with_clusters.csv"
try:
    data.to_csv(output_file, index=False)
    print(f"\nFull labeled dataset saved as '{output_file}'.")
except PermissionError:
    alt_file = "labeled_output_with_clusters_backup.csv"
    data.to_csv(alt_file, index=False)
    print(f"\nPermissionError: Saved the dataset to '{alt_file}' instead.")

# Save the model and encoders
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(label_encoders, "categorical_encoders.pkl")
