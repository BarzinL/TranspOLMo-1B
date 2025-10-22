"""Build datasets for interpretability analysis."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, List, Dict
from tqdm import tqdm


class ActivationDataset(Dataset):
    """Dataset wrapper for tokenized samples."""

    def __init__(self, samples: List[Dict]):
        """
        Initialize dataset.

        Args:
            samples: List of tokenized samples
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class AnalysisDatasetBuilder:
    """Build datasets for interpretability analysis."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        num_samples: int = 10000,
        max_seq_length: int = 512,
        subset: Optional[str] = None,
        split: str = "train"
    ):
        """
        Initialize dataset builder.

        Args:
            dataset_name: Name of HuggingFace dataset
            tokenizer: Tokenizer to use
            num_samples: Number of samples to collect
            max_seq_length: Maximum sequence length
            subset: Dataset subset/configuration
            split: Dataset split to use
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.subset = subset
        self.split = split

    def build(self, batch_size: int = 16, num_workers: int = 0) -> DataLoader:
        """
        Build dataloader for analysis.

        Args:
            batch_size: Batch size for dataloader
            num_workers: Number of worker processes

        Returns:
            DataLoader with tokenized samples
        """
        print(f"Building dataset from {self.dataset_name}...")
        print(f"Target samples: {self.num_samples}")

        # Load dataset
        try:
            if self.subset:
                dataset = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split,
                    streaming=True,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=True,
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to simple text samples...")
            return self._build_fallback_dataset(batch_size, num_workers)

        # Sample and tokenize
        samples = []
        with tqdm(total=self.num_samples, desc="Tokenizing samples") as pbar:
            for i, example in enumerate(dataset):
                if i >= self.num_samples:
                    break

                # Extract text from example (handle different formats)
                text = self._extract_text(example)

                if not text or len(text.strip()) < 10:
                    continue

                # Tokenize
                try:
                    tokens = self.tokenizer(
                        text,
                        max_length=self.max_seq_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )

                    samples.append({
                        'input_ids': tokens['input_ids'].squeeze(0),
                        'attention_mask': tokens['attention_mask'].squeeze(0),
                        'text': text[:500]  # Store truncated text for reference
                    })

                    pbar.update(1)

                except Exception as e:
                    print(f"Error tokenizing sample {i}: {e}")
                    continue

        print(f"Collected {len(samples)} samples")

        # Create dataset and dataloader
        dataset = ActivationDataset(samples)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return dataloader

    def _extract_text(self, example: Dict) -> str:
        """Extract text from dataset example (handles various formats)."""
        # Try common text fields
        for field in ['text', 'content', 'document', 'passage', 'sentence']:
            if field in example:
                return str(example[field])

        # If no common field found, convert whole example to string
        return str(example)

    def _build_fallback_dataset(self, batch_size: int, num_workers: int) -> DataLoader:
        """Build a simple fallback dataset with example texts."""
        print("Creating fallback dataset with example texts...")

        example_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was nothing but darkness and void.",
            "Machine learning is a subset of artificial intelligence.",
            "The capital of France is Paris, a beautiful city on the Seine.",
            "Python is a high-level programming language known for its simplicity.",
            "The universe is estimated to be about 13.8 billion years old.",
            "Climate change is one of the most pressing issues of our time.",
            "The human brain contains approximately 86 billion neurons.",
            "William Shakespeare wrote many famous plays including Hamlet and Macbeth.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        ] * (self.num_samples // 10 + 1)

        samples = []
        for text in example_texts[:self.num_samples]:
            tokens = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            samples.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'text': text
            })

        dataset = ActivationDataset(samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    @staticmethod
    def build_diverse_dataset(
        tokenizer,
        num_samples: int = 10000,
        max_seq_length: int = 512,
        batch_size: int = 16
    ) -> DataLoader:
        """
        Build diverse dataset from multiple sources.

        Args:
            tokenizer: Tokenizer to use
            num_samples: Total number of samples
            max_seq_length: Maximum sequence length
            batch_size: Batch size for dataloader

        Returns:
            DataLoader with diverse samples
        """
        samples_per_source = num_samples // 5

        sources = [
            ('allenai/dolma', 'cc_en_head'),
            ('allenai/dolma', 'stack'),
            ('allenai/dolma', 'pes2o'),
            ('allenai/dolma', 'wiki'),
            ('allenai/c4', 'en'),
        ]

        all_samples = []

        for dataset_name, subset in sources:
            print(f"\nLoading from {dataset_name}/{subset}...")

            builder = AnalysisDatasetBuilder(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                num_samples=samples_per_source,
                max_seq_length=max_seq_length,
                subset=subset
            )

            try:
                loader = builder.build(batch_size=batch_size)
                # Extract samples from loader
                for batch in loader:
                    for i in range(len(batch['input_ids'])):
                        all_samples.append({
                            'input_ids': batch['input_ids'][i],
                            'attention_mask': batch['attention_mask'][i],
                            'text': batch['text'][i] if 'text' in batch else ''
                        })
            except Exception as e:
                print(f"Error loading {dataset_name}/{subset}: {e}")
                continue

        print(f"\nTotal diverse samples collected: {len(all_samples)}")

        dataset = ActivationDataset(all_samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for diversity
            num_workers=0
        )
