coords

# coords[:5][:]

# # print(train_labels.shape)

# print(train_features)

# # Convert to numpy arrays
# train_features = np.array(train_features)
# train_labels = np.array(train_labels)

# # Debugging: print shapes of train_features and train_labels
# print("Training features shape:", train_features.shape)
# print("Training labels shape:", train_labels.shape)

# # Check if train_features and train_labels are populated
# if train_features.size == 0 or train_labels.size == 0:
#     raise ValueError("Training features or labels are empty. Check the labeled data and the TIFF image.")

# # Convert to numpy arrays
# train_features = np.array(train_features)
# train_labels = np.array(train_labels)

# # Train the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(train_features, train_labels)

# # Prepare the entire image data for prediction
# image_features = np.stack([tiff_image[0].ravel(), tiff_image[1].ravel(), tiff_image[2].ravel()], axis=1)

# # Classify the image
# predicted_labels = clf.predict(image_features)

# # # Save the classified image
# # classified_tiff_path = '/mnt/data/Classified_Extract1_tif11.tif'
# # profile.update(dtype=rasterio.uint8, count=1)
# # with rasterio.open(classified_tiff_path, 'w', **profile) as dst:
# #     dst.write(classified_image, 1)

# # Display the classified image
# plt.imshow(classified_image, cmap='tab20')
# plt.colorbar()
# plt.title('Classified Image')
# plt.show()