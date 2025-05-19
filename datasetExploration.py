# Cell 1: Import libraries
import os
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords, wordnet
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import random

os.makedirs('figures', exist_ok=True)

# Cell 2: Set paths and configurations
train_jsonl = 'data/train.jsonl'  
dev_jsonl = 'data/dev.jsonl'      
img_dir = 'data/img/'            

# Cell 3: Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Cell 4: Basic JSONL loading and exploration
print("Loading dataset...")
data = []
with jsonlines.open(train_jsonl) as reader:
    for obj in reader:
        data.append(obj)

# Convert to DataFrame
df = pd.DataFrame(data)

# Display basic information
print(f"Dataset contains {len(df)} memes")
print(f"Columns: {df.columns.tolist()}")
print("\nSample entries:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Cell 5: Explore class distribution
# Class distribution
plt.figure(figsize=(10, 6))
class_counts = df['label'].value_counts().sort_index()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.xticks([0, 1], ['Non-hateful (0)', 'Hateful (1)'], fontsize=10)
plt.ylabel('Count', fontsize=12)

# Add count labels
for i, count in enumerate(class_counts.values):
    plt.text(i, count + 20, str(count), ha='center', fontsize=12)

plt.grid(True, alpha=0.3)
plt.savefig('figures/class_distribution.png')
plt.show()

print("\nClass distribution:")
print(class_counts)
if len(class_counts) > 1:
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Majority class: {'Non-hateful' if class_counts.idxmax() == 0 else 'Hateful'}")
    print(f"Minority class: {'Non-hateful' if class_counts.idxmin() == 0 else 'Hateful'}")

# Cell 6: Text length analysis
# Analyze text length
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\nText length statistics (characters):")
print(f"Min: {df['text_length'].min()}")
print(f"Max: {df['text_length'].max()}")
print(f"Mean: {df['text_length'].mean():.2f}")
print(f"Median: {df['text_length'].median()}")

print("\nWord count statistics:")
print(f"Min: {df['word_count'].min()}")
print(f"Max: {df['word_count'].max()}")
print(f"Mean: {df['word_count'].mean():.2f}")
print(f"Median: {df['word_count'].median()}")

# Plot text length distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Length (Characters)')
plt.xlabel('Number of Characters')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(df['word_count'], bins=30, kde=True)
plt.title('Distribution of Word Count')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/text_length_distribution.png')
plt.show()

# Compare text length between classes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='label', y='text_length', data=df)
plt.title('Text Length by Class')
plt.xlabel('Class (0=Non-hateful, 1=Hateful)')
plt.ylabel('Number of Characters')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(x='label', y='word_count', data=df)
plt.title('Word Count by Class')
plt.xlabel('Class (0=Non-hateful, 1=Hateful)')
plt.ylabel('Number of Words')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/text_length_by_class.png')
plt.show()

# Cell 7: Display sample memes
# Display sample memes (both hateful and non-hateful)
def display_samples(df, img_dir, num_per_class=3):
    """Display sample memes from each class"""
    # Get samples from each class
    hateful = df[df['label'] == 1].sample(min(num_per_class, len(df[df['label'] == 1])))
    non_hateful = df[df['label'] == 0].sample(min(num_per_class, len(df[df['label'] == 0])))
    
    total_samples = len(hateful) + len(non_hateful)
    plt.figure(figsize=(15, 5 * total_samples))
    
    # Display non-hateful samples
    for i, (_, row) in enumerate(non_hateful.iterrows()):
        img_path = os.path.join(img_dir, row['img'])
        
        plt.subplot(total_samples, 1, i+1)
        try:
            img = Image.open(img_path).convert('RGB')
            plt.imshow(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            plt.text(0.5, 0.5, f"Error loading image: {e}", ha='center', va='center')
        
        plt.title(f"Non-hateful Meme - Text: {row['text']}")
        plt.axis('off')
    
    # Display hateful samples
    for i, (_, row) in enumerate(hateful.iterrows()):
        img_path = os.path.join(img_dir, row['img'])
        
        plt.subplot(total_samples, 1, i+len(non_hateful)+1)
        try:
            img = Image.open(img_path).convert('RGB')
            plt.imshow(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            plt.text(0.5, 0.5, f"Error loading image: {e}", ha='center', va='center')
        
        plt.title(f"Hateful Meme - Text: {row['text']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/sample_memes.png')
    plt.show()

print("\nDisplaying sample memes...")
display_samples(df, img_dir, num_per_class=3)

# Cell 8: Text analysis with word clouds
# Create word clouds for all text
print("\nCreating word clouds...")

# For all memes
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()
importance = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
word_importance = dict(zip(feature_names, importance))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100,
    colormap='viridis'
).generate_from_frequencies(word_importance)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Important Words in All Memes', fontsize=16)
plt.savefig('figures/all_wordcloud.png')
plt.show()

# Cell 9: Separate word clouds by class
# Create separate word clouds for each class
for label, name in [(0, 'Non-hateful'), (1, 'Hateful')]:
    class_texts = df[df['label'] == label]['text']
    
    if len(class_texts) == 0:
        print(f"No texts found for class {name}")
        continue
        
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(class_texts)
    feature_names = vectorizer.get_feature_names_out()
    importance = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    word_importance = dict(zip(feature_names, importance))
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='plasma' if label == 1 else 'viridis'
    ).generate_from_frequencies(word_importance)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Important Words in {name} Memes', fontsize=16)
    plt.savefig(f'figures/{name.lower()}_wordcloud.png')
    plt.show()

# Cell 10: Define the Dataset class
# Custom PyTorch Dataset for Hateful Memes
class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, transform=None, 
                 tokenizer=None, max_len=128, text_processor='bert',
                 augment=False, split='train'):
        """
        Custom PyTorch Dataset for the Hateful Memes dataset
        
        Args:
            jsonl_file (str): Path to the metadata JSONL file
            img_dir (str): Directory with all the images
            transform (callable, optional): Transform to apply to the images
            tokenizer: Tokenizer to use for text processing
            max_len (int): Maximum sequence length for tokenization
            text_processor (str): 'bert' or 'lstm' for different text processing
            augment (bool): Whether to apply data augmentation
            split (str): 'train', 'val', or 'test' split
        """
        # Load the JSONL data
        data = []
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                data.append(obj)
                
        # Convert to DataFrame
        self.data = pd.DataFrame(data)
        
        self.img_dir = img_dir
        self.split = split
        self.max_len = max_len
        self.text_processor = text_processor
        self.augment = augment and split == 'train'  # Only augment training data
        
        # Initialize tokenizer for BERT
        if tokenizer is None and text_processor == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        # Set up image transformations
        if transform is None:
            # Standard preprocessing for ResNet/CNN
            if self.augment:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])  # ImageNet stats
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])  # ImageNet stats
                ])
        else:
            self.transform = transform
        
        # Load stopwords for text preprocessing
        self.stop_words = set(stopwords.words('english'))
        
        # Count class distribution
        class_counts = self.data['label'].value_counts()
        print(f"Class distribution in {split} set: {class_counts.to_dict()}")
        
        # Compute weights for balanced sampling
        self.weights = self._compute_sample_weights()
    
    def _compute_sample_weights(self):
        """Compute weights for each sample to handle class imbalance"""
        class_counts = self.data['label'].value_counts()
        n_samples = len(self.data)
        n_classes = len(class_counts)
        
        # Calculate weight for each class
        weights = n_samples / (n_classes * class_counts)
        
        # Assign weight to each sample based on its class
        sample_weights = [weights[label] for label in self.data['label']]
        
        return torch.FloatTensor(sample_weights)
    
    def _augment_text(self, text):
        """Simple text augmentation via synonym replacement"""
        words = text.split()
        if len(words) <= 3:  # Don't augment very short texts
            return text
               
        num_to_replace = max(1, int(len(words) * 0.2))  # Replace 20% of words
        replace_indices = random.sample(range(len(words)), num_to_replace)
        
        for idx in replace_indices:
            word = words[idx]
            # Skip short words and non-alpha words
            if len(word) <= 3 or not word.isalpha():
                continue
                
            # Get synonyms
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        synonyms.append(synonym)
                        
            if synonyms:
                words[idx] = random.choice(synonyms)
                
        return ' '.join(words)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get data for this sample
        sample = self.data.iloc[idx]
        
        # Get image path and load image
        img_path = os.path.join(self.img_dir, sample['img'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Apply transformations to image
        if self.transform:
            image = self.transform(image)
            
        # Get text and apply augmentation if enabled
        text = sample['text']
        
        if self.augment and random.random() < 0.5:  # 50% chance of augmentation
            text = self._augment_text(text)
            
        # Process text based on specified processor
        if self.text_processor == 'bert':
            # Tokenize for BERT
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            processed_text = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:  # LSTM processing
            # Tokenize and remove stopwords
            tokens = [token.lower() for token in text.split() 
                     if token.lower() not in self.stop_words]
            
            # For now, just return the tokens - we'll handle embeddings later
            processed_text = tokens
        
        # Get the label
        label = torch.tensor(sample['label'])
        
        return {
            'image': image,
            'text': processed_text,
            'label': label,
            'raw_text': text  # Keep raw text for visualization
        }

# Cell 11: Define the DataLoader function
def get_dataloaders(jsonl_train, jsonl_val, img_dir, batch_size=32, 
                   text_processor='bert', augment=True, num_workers=4):
    """
    Create train and validation DataLoaders
    
    Args:
        jsonl_train (str): Path to training JSONL file
        jsonl_val (str): Path to validation JSONL file
        img_dir (str): Directory with images
        batch_size (int): Batch size
        text_processor (str): 'bert' or 'lstm'
        augment (bool): Whether to apply augmentation
        num_workers (int): Number of worker threads
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = HatefulMemesDataset(
        jsonl_file=jsonl_train,
        img_dir=img_dir,
        text_processor=text_processor,
        augment=augment,
        split='train'
    )
    
    val_dataset = HatefulMemesDataset(
        jsonl_file=jsonl_val,
        img_dir=img_dir,
        text_processor=text_processor,
        augment=False,  # No augmentation for validation
        split='val'
    )
    
    # Create samplers for handling class imbalance
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

# Cell 12: Test the Dataset and DataLoader
# Test the Dataset and DataLoader
print("\nTesting Dataset and DataLoader...")

# Create train and validation datasets
train_dataset = HatefulMemesDataset(
    jsonl_file=train_jsonl,
    img_dir=img_dir,
    text_processor='bert',
    augment=False,  # No augmentation for testing
    split='train'
)

val_dataset = HatefulMemesDataset(
    jsonl_file=dev_jsonl,
    img_dir=img_dir,
    text_processor='bert',
    augment=False,
    split='val'
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Test retrieving an item
sample = train_dataset[0]
print("\nSample item from dataset:")
print(f"Image shape: {sample['image'].shape}")
print(f"Text: {sample['raw_text']}")
print(f"Label: {sample['label'].item()}")
print(f"BERT input_ids shape: {sample['text']['input_ids'].shape}")
print(f"BERT attention_mask shape: {sample['text']['attention_mask'].shape}")

# Cell 13: Test DataLoader with weighted sampling
# Create DataLoaders
train_loader, val_loader, _, _ = get_dataloaders(
    jsonl_train=train_jsonl,
    jsonl_val=dev_jsonl,
    img_dir=img_dir,
    batch_size=32,
    text_processor='bert',
    augment=True,
    num_workers=4
)

print(f"\nTrain loader batches: {len(train_loader)}")
print(f"Validation loader batches: {len(val_loader)}")

# Get a batch from the train loader
batch = next(iter(train_loader))
print("\nBatch information:")
print(f"Batch size: {len(batch['label'])}")
print(f"Image tensor shape: {batch['image'].shape}")
print(f"Text input_ids shape: {batch['text']['input_ids'].shape}")
print(f"Text attention_mask shape: {batch['text']['attention_mask'].shape}")
print(f"Label shape: {batch['label'].shape}")

# Check class balance in the batch (should be roughly balanced due to weighted sampling)
label_counts = batch['label'].cpu().numpy()
unique, counts = np.unique(label_counts, return_counts=True)
class_counts_batch = dict(zip(unique, counts))

plt.figure(figsize=(8, 6))
sns.barplot(x=list(class_counts_batch.keys()), y=list(class_counts_batch.values()))
plt.title('Class Distribution in Batch (with Weighted Sampling)')
plt.xlabel('Class')
plt.xticks([0, 1], ['Non-hateful', 'Hateful'])
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.savefig('figures/batch_class_distribution.png')
plt.show()

print("\nClass distribution in batch:")
print(class_counts_batch)

# Cell 14: Test text augmentation
# Test text augmentation
def test_text_augmentation():
    """Test the text augmentation function with some examples"""
    # Get some sample texts
    sample_texts = df['text'].sample(5).tolist()
    
    print("\nText Augmentation Examples:")
    for i, text in enumerate(sample_texts):
        # Create a simple augmentation function (similar to the one in the Dataset class)
        def augment_text(text):
            words = text.split()
            if len(words) <= 3:
                return text
                
            num_to_replace = max(1, int(len(words) * 0.2))
            replace_indices = random.sample(range(len(words)), num_to_replace)
            
            for idx in replace_indices:
                word = words[idx]
                if len(word) <= 3 or not word.isalpha():
                    continue
                    
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != word:
                            synonyms.append(synonym)
                            
                if synonyms:
                    words[idx] = random.choice(synonyms)
                    
            return ' '.join(words)
        
        # Apply augmentation
        augmented = augment_text(text)
        
        # Print original and augmented
        print(f"Original [{i+1}]: {text}")
        print(f"Augmented [{i+1}]: {augmented}")
        print()

# Test the augmentation
test_text_augmentation()

# Cell 15: Visualize a batch of images
# Visualize a batch of images
def visualize_batch(batch, num_images=8):
    """Visualize a batch of images"""
    images = batch['image'][:num_images].cpu()
    labels = batch['label'][:num_images].cpu().numpy()
    texts = batch['raw_text'][:num_images]
    
    # Create a grid of images
    plt.figure(figsize=(20, 10))
    for i in range(min(num_images, len(images))):
        # Convert tensor to image
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(f"Label: {'Hateful' if labels[i] == 1 else 'Non-hateful'}")
        plt.xlabel(f"Text: {texts[i]}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('figures/batch_visualization.png')
    plt.show()

# Get a batch and visualize
batch = next(iter(train_loader))
visualize_batch(batch)

# Cell 16: Summary of findings
# Summarize dataset characteristics
print("\n===== DATASET SUMMARY =====")
print(f"Total number of memes: {len(df)}")
print(f"Number of hateful memes: {len(df[df['label'] == 1])}")
print(f"Number of non-hateful memes: {len(df[df['label'] == 0])}")
print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
print(f"Average text length: {df['text_length'].mean():.2f} characters")
print(f"Average word count: {df['word_count'].mean():.2f} words")

# Key findings
print("\nKey Findings:")
print("1. The dataset consists of memes with both images and text components")
print(f"2. Class distribution shows {'balance' if imbalance_ratio < 1.2 else 'imbalance'} between hateful and non-hateful memes")
print("3. Text length and word count distributions provide insights for tokenization limits")
print("4. Word clouds highlight the most important terms in each class")
print("5. Data augmentation techniques can help address potential class imbalance")
print("6. The PyTorch Dataset and DataLoader implementation successfully loads and processes the data")

print("\nNext Steps:")
print("1. Implement text models (LSTM, BERT) for processing meme text")
print("2. Implement image models (CNN, ResNet) for processing meme images")
print("3. Develop late fusion strategies to combine modalities")
print("4. Develop early fusion strategies for comparison")
print("5. Set up evaluation metrics and TensorBoard logging")

print("\nDataset exploration complete!")