import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import get_args_parser, cal_contrastiveloss

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def visualize_features(features, labels, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method {method}")

    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f'Feature Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    # Load the best model
    checkpoint = torch.load(args.output_dir + '/checkpoint-best.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Extract features
    features, labels = extract_features(model, current_val_loader, device)

    # Visualize features
    visualize_features(features, labels, method='pca')
