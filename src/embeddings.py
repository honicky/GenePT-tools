import json
from pathlib import Path
import pandas as pd
from openai import OpenAI

class BatchInfo:
    """
    A class to manage batch processing information for OpenAI API requests.

    This class handles the organization of file paths and metadata for batch processing
    of API requests and their corresponding responses.

    Args:
        batch_name (str): Unique identifier for the batch
        request_data (list[dict]): List of request objects to be processed in the batch
        batch_description (str): Human-readable description of the batch's purpose
        data_dir (Path): Base directory path for storing request and response files

    Attributes:
        batch_name (str): Name of the batch
        request_filename (str): Filename for storing batch requests
        request_file_path (str): Full path to the request file
        response_filename (str): Filename for storing batch responses
        response_file_path (str): Full path to the response file
        batch_description (str): Human-readable description of the batch's purpose
        request_data (list[dict]): The actual request data to be processed
    """

    def __init__(self,
        batch_name: str,
        request_data: list[dict],
        batch_description: str,
        data_dir: Path
    ):
        self.batch_name = batch_name
        self.request_filename = f"{batch_name}_requests.jsonl"
        self.request_file_path = str(data_dir / "generated" / "batch-requests" / self.request_filename)
        self.response_filename = f"{batch_name}_responses.jsonl"
        self.response_file_path = str(data_dir / "generated" / "batch-requests" / self.response_filename)
        self.batch_description = batch_description
        self.request_data = request_data
        self.data_dir = data_dir

def get_gene_text_batch_requests(summary_of_genes: dict, prompt_template: str, request_id_prefix: str, model: str = "gpt-4o-mini") -> list[dict]:
    """
    Generate a list of batch request objects for processing gene summaries through OpenAI's API.

    Args:
        summary_of_genes (dict): Dictionary mapping gene names to their summaries
        prompt_template (str): Template string for formatting the user prompt. Should contain
                             two placeholder positions for gene name and summary
        request_id_prefix (str): Prefix to use for generating unique request IDs

    Returns:
        list[dict]: List of request objects formatted for OpenAI's batch processing API. Each request includes:
                   - custom_id: Unique identifier prefix for each request in the batch
                   - method: HTTP method (POST)
                   - url: API endpoint
                   - body: Request body containing model configuration and messages

    Note:
        Each request uses the model with a 2000 token limit.
    """
    return [
        {
            "custom_id": f"{request_id_prefix}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a detail oriented bioinformatics and genetics expert help to give accurate and precise descriptions of genes",
                    },
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            gene, summary_of_genes[gene]
                        ),
                    },
                ],
                "max_tokens": 2000,
            },
        }
        for i, gene in enumerate(list(summary_of_genes.keys()))
    ]


def get_gene_embedding_batch_requests(gene_descriptions_pdf: pd.DataFrame, request_id_prefix: str, model: str = "text-embedding-3-large") -> list[dict]:
    
    prompt_template = """
    {0}
    
    {1}
    """
    
    return [
        {
            "custom_id": f"full-batch-embedding-request-{i}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": prompt_template.format(
                    row.description, row.gpt_response
                ),
                "encoding_format": "float",
            },
        }
        for i, (gene_name, row) in enumerate(gene_descriptions_pdf.iterrows())

    ]


def _create_batch_file(batch_info: BatchInfo, client: OpenAI) -> any:
    """
    Create a batch file for OpenAI API processing from batch information.

    Args:
        batch_info (BatchInfo): Object containing batch processing information and request data
        client (OpenAI): OpenAI client instance for API interactions

    Returns:
        The created file object from the OpenAI API containing the batch requests

    Note:
        This function writes the batch requests to a JSONL file and uploads it to OpenAI
        for batch processing.
    """
    output_path = write_batch_requests_jsonl(batch_info.request_data, batch_info.request_filename, batch_info.data_dir)
    batch_input_filename = client.files.create(
        file=open(output_path, "rb"), purpose="batch"
    )
    return batch_input_filename

def write_batch_requests_jsonl(requests: list, filename: str, data_dir: Path) -> Path:
    """
    Write a list of JSON objects to a JSONL file in the batch-requests directory.

    Args:
        requests: List of JSON objects to write
        filename: Name of the file (without .jsonl extension)
        data_dir: Path object pointing to base data directory
    Returns:
        Path: Path object pointing to the created file
    """
    # Create the directory path
    output_dir = data_dir / "generated" / "batch-requests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure filename has .jsonl extension
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    output_path = output_dir / filename

    # Write the requests line by line
    with open(output_path, "w") as f:
        for request in requests:
            json_line = json.dumps(request)
            f.write(json_line + "\n")

    return output_path

def create_batch_job(batch_info: BatchInfo, type: str, client: OpenAI) -> any:
    """
    Create a batch processing job using the OpenAI API.

    Args:
        batch_info (BatchInfo): Object containing batch processing information and request data
        client (OpenAI): OpenAI client instance for API interactions

    Returns:
        The created batch job object containing processing details and status

    Note:
        This function creates a batch file and initiates a batch processing job with a
        24-hour completion window. The batch job processes chat completions using the
        configuration specified in the batch_info.
    """

    if type == "completion":
        endpoint = "/v1/chat/completions"
    elif type == "embedding":
        endpoint = "/v1/embeddings"
    else:
        raise ValueError(f"Invalid batch type: {type}")
    
    batch_input_file = _create_batch_file(batch_info, client)
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={
            "description": batch_info.batch_description,
        },
    )
    return batch_job

import time

def monitor_batch_status(client, batch, check_interval=60, verbose=True):
    """
    Monitor the status of an OpenAI batch processing job.

    Args:
        client: OpenAI client instance
        batch: Batch object or ID returned from client.batches.create()
        check_interval: Time in seconds between status checks (default: 60)
        verbose: Whether to print progress updates (default: True)

    Returns:
        BatchStatus object for the completed batch
    """
    try:

        if isinstance(batch, str):
            batch_id = batch
        else:
            batch_id = batch.id

        # Monitor completion progress
        while True:
            batch_status = client.batches.retrieve(batch_id)

            completed = batch_status.request_counts.completed
            failed = batch_status.request_counts.failed
            total = batch_status.request_counts.total

            if verbose:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Completed: {completed}, Failed: {failed}, Total: {total}"
                )

            if completed + failed == total:
                break

            time.sleep(check_interval)

        # Wait for final batch completion
        while True:
            batch_status = client.batches.retrieve(batch_id)
            if (
                batch_status.completed_at is not None
                or batch_status.failed_at is not None
            ):
                break
            if verbose:
                print(batch_status)

            time.sleep(check_interval)

        if verbose:
            print("\nBatch completed")

        return batch_status

    except Exception as e:
        print(f"Error monitoring batch status: {str(e)}")
        raise


def save_batch_response(batch_info: BatchInfo, batch_status, client: OpenAI) -> Path:
    """
    Save the response from an OpenAI batch operation to a JSONL file.

    Args:
        batch_info: BatchInfo object containing batch processing information
        batch_status: BatchStatus object from completed batch
        client: OpenAI client instance

    Returns:
        Path: Path object pointing to the saved response file
    """
    # Get the response content
    file_response = client.files.content(batch_status.output_file_id)

    # Create output path
    output_path = Path(batch_info.response_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write response to file
    with open(output_path, "w") as f:
        f.write(file_response.text)

    return output_path


def load_batch_responses(batch_info: BatchInfo) -> list[dict]:
    """
    Load the responses from an OpenAI batch operation from a JSONL file.

    Args:
        batch_info (BatchInfo): Object containing batch processing information and request data

    Returns:
        list[dict]: List of response objects from the batch operation
    """
    responses = []
    with open(batch_info.response_file_path, "r") as f:
        for line in f:
            responses.append(json.loads(line))
    return responses


def create_gene_descriptions_dataframe(
        ncbi_uniprot_summary_of_genes: dict,
        batch_responses: list[dict],
        gene_info_table: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Create a DataFrame containing gene descriptions and their GPT responses.
    
    Args:
        ncbi_uniprot_summary_of_genes (dict): Dictionary mapping gene names to their descriptions
        batch_responses (list): List of batch response objects from the API
        gene_info_table (pd.DataFrame): DataFrame mapping gene names to their ensembl gene ids and type information
        
    Returns:
        pd.DataFrame: DataFrame with gene descriptions and GPT responses, indexed by gene name
    """
    # Create lists to store our data
    gene_names = list(ncbi_uniprot_summary_of_genes.keys())
    descriptions = list(ncbi_uniprot_summary_of_genes.values())
    gpt_responses = [
        response["response"]["body"]["choices"][0]["message"]["content"]
        for response in batch_responses
    ]

    # Create the DataFrame
    gene_descriptions_pdf = pd.DataFrame(
        {"description": descriptions, "gpt_response": gpt_responses}, 
        index=gene_names
    )

    # Rename the index
    gene_descriptions_pdf.index.name = "gene_name"
    
    # Merge the gene descriptions with the gene info table
    gene_descriptions_pdf = gene_descriptions_pdf.merge(
        gene_info_table,
        left_index=True,
        right_index=True,
        how="left"
    )

    return gene_descriptions_pdf

