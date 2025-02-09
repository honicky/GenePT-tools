# Tabula Sapiens 1M Cells Dataset Metadata Description

This document describes the metadata columns available in the 1M cells dataset's observation (cell) annotations.

## Experimental/Technical Information
- `10X_run`: Identifier for the 10X Genomics sequencing run
- `cdna_plate`, `cdna_well`: Physical location of the sample in the cDNA preparation plate
- `library_plate`: Plate identifier for library preparation
- `method`: Experimental method used for cell isolation/sequencing
- `assay`, `assay_ontology_term_id`: Type of assay used and its ontology ID
- `suspension_type`: How the cells were suspended for analysis

## Quality Control Metrics
- `total_counts`: Total number of RNA molecules counted in this cell
- `n_genes_by_counts`: Number of genes detected in this cell
- `pct_counts_mt`: Percentage of counts from mitochondrial genes
- `pct_counts_ercc`: Percentage of counts from ERCC spike-in controls
- `total_counts_ercc`: Total counts from ERCC spike-in controls
- `total_counts_mt`: Total counts from mitochondrial genes
- `ambient_removal`: Whether ambient RNA contamination was removed

## Cell Identity
- `cell_type`, `cell_type_ontology_term_id`: Annotated cell type and its ontology ID
- `broad_cell_class`: Broader classification of cell type
- `manually_annotated`: Whether the cell type was manually annotated

## Biological Context
- `anatomical_position`: Location in the body where the cell was from
- `tissue`, `tissue_type`, `tissue_ontology_term_id`: Tissue source and its ontology ID
- `tissue_in_publication`: How the tissue was referred to in publication
- `compartment`: Anatomical compartment the cell was from
- `development_stage`, `development_stage_ontology_term_id`: Developmental stage and its ontology ID

## Donor Information
- `donor_id`: Unique identifier for the donor
- `donor_tissue`: Tissue provided by the donor
- `donor_method`: Method used to obtain the sample from donor
- `donor_assay`, `donor_tissue_assay`: Specific assay details related to the donor sample
- `ethnicity_original`, `self_reported_ethnicity`, `self_reported_ethnicity_ontology_term_id`: Donor ethnicity information
- `sex`, `sex_ontology_term_id`: Donor sex and its ontology ID
- `disease`, `disease_ontology_term_id`: Any disease condition and its ontology ID

## Dataset Organization
- `_index`: Unique identifier for each cell
- `observation_joinid`: ID used to join different datasets
- `sample_id`, `sample_number`: Sample identifiers
- `replicate`: Replicate number for the experiment
- `is_primary_data`: Whether this is primary or derived data
- `published_2022`: Whether this data was published in 2022

## Analysis-specific
- `_scvi_batch`, `_scvi_labels`: Labels used in scVI analysis (a deep learning method for single-cell data)
- `scvi_leiden_donorassay_full`: Clustering results using the Leiden algorithm
- `free_annotation`: Free-text annotations
- `notes`: Additional notes about the cell

## Metadata
- `organism`, `organism_ontology_term_id`: Source organism and its ontology ID


# Gene (Variable) Metadata Description

## Gene Identifiers
- `ensembl_id`: Ensembl database identifier for the gene
- `ensg`: Ensembl gene ID (ENSG format)
- `feature_name`: Common name or symbol of the gene
- `feature_reference`: Reference database or source of the feature annotation

## Gene Characteristics
- `feature_biotype`: Biological type of the feature (e.g., protein coding, lncRNA, etc.)
- `feature_length`: Length of the feature in base pairs
- `feature_type`: Type of genomic feature
- `genome`: Reference genome used for annotation
- `mt`: Boolean indicating if the gene is mitochondrial
- `ercc`: Boolean indicating if the feature is an ERCC spike-in control

## Quality Control and Statistics
- `feature_is_filtered`: Boolean indicating if the feature was filtered out in preprocessing
- `mean`: Mean expression value across all cells
- `mean_counts`: Average count of this feature across all cells
- `n_cells_by_counts`: Number of cells where this feature is detected
- `pct_dropout_by_counts`: Percentage of cells where this feature has zero counts
