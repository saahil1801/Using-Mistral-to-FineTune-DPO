from datasets import Dataset, load_dataset
import warnings
warnings.filterwarnings("ignore")
import logging
from config import Config





def dpo_data(dataset_id, split='train_prefs') -> Dataset :
    logging.info(f'Loading dataset {dataset_id} with split {split}')
    # Load the dataset
    dataset = load_dataset(dataset_id, split=split, use_auth_token=False)

    # Function to retain only necessary columns
    def simplify_record(samples):
        logging.debug('Simplifying record')
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"]
        }

    # Apply the simplification and remove original columns
    processed_dataset = dataset.map(simplify_record, batched=True, remove_columns=dataset.column_names)
    return processed_dataset

def create_dataset(dataset_id, split='train_prefs'):
    logging.info(f'Creating dataset from {dataset_id} with split {split}')
    # Load and preprocess the dataset
    dataset = dpo_data(dataset_id, split)

    # Convert to pandas DataFrame for further processing
    df = dataset.to_pandas()

    # Extract content from the 'chosen' and 'rejected' columns
    df["chosen"] = df["chosen"].apply(lambda x: x[1]["content"])
    df["rejected"] = df["rejected"].apply(lambda x: x[1]["content"])

    logging.debug('Extracted content from chosen and rejected columns')

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Convert back to Hugging Face's Dataset format
    final_dataset = Dataset.from_pandas(df)
    return final_dataset

if __name__ == '__main__':
    config = Config()
    database_id = config.DATASET_ID

    logging.info(f'Start processing dataset {database_id}')

    dataset = create_dataset(database_id, split='train_prefs')
    logging.info(f'Finished processing dataset {database_id}')
    print(dataset)
