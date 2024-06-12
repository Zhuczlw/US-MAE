from lib import *

class CustomImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if root_dir:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.classes = None
        self.images = self._load_images()
        self.class_counts = self._count_images_per_class()

    def _load_images(self):
        images = []
        if self.classes == None:
            return
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                images.append((img_path, self.class_to_idx[class_name]))
        return images

    def _count_images_per_class(self):
        if self.classes == None:
            return
        class_counts = {cls_name: 0 for cls_name in self.classes}
        for _, label in self.images:
            class_counts[self.classes[label]] += 1
        return class_counts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        # image = datasets.folder.default_loader(img_path)
        # if self.transform:
        #     image = self.transform(image)
        # return image, label
        # try:
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
        # except:
        #     os.remove(img_path)
        #     print(f"Deleted image: {img_path}")
        #     return None, None


class ConcatCustomImageFolderDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_length = len(dataset1) + len(dataset2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

def split_dataset(root_dir, n_tasks, train_batch_size=48, test_batch_size=64, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = CustomImageFolderDataset(root_dir+'train', transform=transform)
    val_dataset = CustomImageFolderDataset(root_dir+'val', transform=transform)

    # Randomly shuffle the dataset indices
    # rng = np.random.default_rng(seed)
    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(val_dataset))

    # rng.shuffle(indices)

    # Split dataset into tasks
    train_task_size = len(train_dataset.class_counts) // n_tasks
    val_task_size = len(val_dataset.class_counts) // n_tasks
    a = list(train_dataset.class_counts.values())
    b = list(val_dataset.class_counts.values())
    train_task_idx = [list(train_indices[sum(a[:train_task_size*i]):sum(a[:train_task_size*(i+1)])])  for i in range(n_tasks)]
    train_task_idx_temp = train_task_idx
    val_task_idx = [val_indices[sum(b[:val_task_size*i]):sum(b[:val_task_size*(i+1)])]  for i in range(n_tasks)]
    train_task_datasets = [Subset(train_dataset, train_indices[sum(a[:train_task_size*i]):sum(a[:train_task_size*(i+1)])])  for i in range(n_tasks)] #单纯每个任务的数据，不包含增量
    val_task_datasets = [Subset(val_dataset, val_indices[sum(b[:val_task_size*i]):sum(b[:val_task_size*(i+1)])])  for i in range(n_tasks)]

    train_Inc_task_idx = [random.sample(list(train_task_idx[i]), 500) for i in range(n_tasks)]
    train_Inc_task_idx_no_add = [sum(train_Inc_task_idx[:i + 1], []) for i in range(len(train_Inc_task_idx))][:-1]
    train_Inc_task_idx_no_add.insert(0, []) #增量的
    [y.extend(x) for x, y in zip(train_Inc_task_idx_no_add, train_task_idx_temp)] #增量的和需要训练的加起来
    train_Inc_task_idx_all = train_task_idx_temp ##更新


    val_task_idx_idx = [list(val_task_idx[i]) for i in range(n_tasks)]
    val_Inc_task_idx_all = [sum(val_task_idx_idx[:i + 1], []) for i in range(len(val_task_idx_idx))]

    train_task_datasets = [Subset(train_dataset, train_Inc_task_idx_all[i]) for i in range(len(train_Inc_task_idx_all))]
    train_Inc_task_datasets = [Subset(train_dataset, train_Inc_task_idx_no_add[i]) for i in range(len(train_Inc_task_idx_no_add))]
    train_all_task_datasets = [Subset(train_dataset, train_Inc_task_idx_all[i]) for i in
                               range(len(train_Inc_task_idx_all))]

    val_Inc_task_datasets = [Subset(val_dataset, val_Inc_task_idx_all[i]) for i in range(len(val_Inc_task_idx_all))]

    # DataLoader for each task
    train_loaders = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in train_task_datasets]
    train_Inc_loaders = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in train_Inc_task_datasets[1:]]
    train_Inc_loaders.insert(0,[])
    train_all_loaders = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in
                         train_all_task_datasets]
    test_loaders = [DataLoader(dataset, batch_size=test_batch_size, shuffle=False) for dataset in val_Inc_task_datasets]

    return (train_task_datasets, val_task_datasets, train_loaders, train_Inc_loaders, train_all_loaders, test_loaders)
            # , train_Inc_loaders, test_Inc_loaders)

def multi_dataset(args, root_dir_list, n_tasks, train_batch_size=64, test_batch_size=64, seed=0, Inc_img_nums = 500):
    random.seed(seed)
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    assert n_tasks == len(root_dir_list)
    train_datasets = [ImageFolder(root_dir + 'train', transform=transform) for root_dir in root_dir_list]
    val_datasets = [ImageFolder(root_dir + 'val', transform=transform) for root_dir in root_dir_list]
    index_add_list = [sum(len(dataset.classes) for dataset in train_datasets[:i]) for i in
                      range(len(train_datasets) + 1)][:-1]
    for i, dataset in enumerate(train_datasets):
        index_offset = index_add_list[i]  # Get the offset value for this dataset
        for j in range(len(dataset)):  # Iterate over each sample in the dataset
            old_index = dataset.targets[j]  # Get the original index
            new_index = old_index + index_offset  # Adjust the index using the offset
            dataset.targets[j] = new_index  # Update the index in the dataset
            dataset.samples[j] = (dataset.samples[j][0], new_index)  # Update the sample with new index
    for i, dataset in enumerate(val_datasets):
        index_offset = index_add_list[i]  # Get the offset value for this dataset
        for j in range(len(dataset)):  # Iterate over each sample in the dataset
            old_index = dataset.targets[j]  # Get the original index
            new_index = old_index + index_offset  # Adjust the index using the offset
            dataset.targets[j] = new_index  # Update the index in the dataset
            dataset.samples[j] = (dataset.samples[j][0], new_index)  # Update the sample with new index

    # Group samples of each category
    class_to_images = [defaultdict(list) for _ in range(len(train_datasets))]
    for i, dataset in enumerate(train_datasets):
        for img_path, label in dataset.samples:
            class_to_images[i][label-index_add_list[i]].append((img_path, label))

    # Select 500 samples from each category from different datasets
    train_Inc_datasets = []
    for task_id in range(len(class_to_images)):
        train_Inc_dataset = []
        for class_id in range(len(class_to_images[task_id])):
            class_images = class_to_images[task_id][class_id]
            random.shuffle(class_images)
            subset = class_images[:Inc_img_nums]
            subset_dataset = CustomImageFolderDataset(root_dir='', transform=transform)
            subset_dataset.images = subset
            train_Inc_dataset = ConcatCustomImageFolderDataset(train_Inc_dataset, subset_dataset)
        train_Inc_datasets.append(train_Inc_dataset)

    train_Inc_datasets.insert(0, [])
    train_Inc_datasets = train_Inc_datasets[:-1]
    train_all_datasets = [ConcatCustomImageFolderDataset(x,y) for x, y in zip(train_Inc_datasets, train_datasets)] #增量的和需要训练的加起来
    current_Inc_loader = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in train_Inc_datasets[1:]]
    train_all_loaders = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in train_all_datasets]
    train_loaders = [DataLoader(dataset, batch_size=train_batch_size, shuffle=True) for dataset in train_datasets]
    current_Inc_loader.insert(0,[])
    test_split_loaders = [DataLoader(dataset, batch_size=test_batch_size, shuffle=False) for dataset in val_datasets]
    accumulated_datasets = []
    accumulated_data = None
    for dataset in val_datasets:
        if accumulated_data is None:
            accumulated_data = dataset
        else:
            accumulated_data = torch.utils.data.ConcatDataset([accumulated_data, dataset])
        accumulated_datasets.append(accumulated_data)
    val_datasets = accumulated_datasets
    test_loaders = [DataLoader(dataset, batch_size=test_batch_size, shuffle=False) for dataset in val_datasets]
    return train_all_datasets, val_datasets, train_all_loaders, current_Inc_loader, test_loaders, train_loaders, test_split_loaders


