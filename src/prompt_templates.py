
NCBI_UNIPROT_ASSOCIATED_AGEING_DRUG_PATHWAY_PROMPT_V1 = """Tell me about the {0} gene.

Here is the NCBI and UniProt summary of the gene:

{1}

----

In addition to the provided information, please:

1. List any other genes that the gene is associated with, particularly those not mentioned in the summaries above.
2. Highlight any known details related to aging or age-related diseases that might be associated with this gene.
3. List any drug or drug classes that are known to interact with this gene. 
4. Pathways and biological processes that this gene is involved in.
"""

NCBI_UNIPROT_ASSOCIATED_CELL_TYPE_DRUG_PATHWAY_PROMPT_V1 = """Tell me about the {0} gene.

Here is the NCBI and UniProt summary of the gene:

{1}

----

In addition to the provided information, please:

1. List any other genes that the gene is associated with, particularly those not mentioned in the summaries above.
2. List any cell types or cell classes that the gene is expressed in.
3. List any drug or drug classes that are known to interact with this gene. 
4. Pathways and biological processes that this gene is involved in.

Only include specific information about the gene or gene class. If information is not well documented, say so briefly and don't expound on general information.
"""
