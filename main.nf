#!/usr/bin/env nextflow

/*
 * MERIT v2.0 — MastEr Regulator Inference Tool
 *
 * Nextflow pipeline for context-specific gene regulatory network inference.
 *
 * Pipeline steps:
 *   1. GRN inference (ensemble: RF + LASSO + CLR with Monte Carlo thresholding)
 *   2. TFBS filtering (PWM motif scanning or CNN-based)
 *   3. Context-specific annotation (treatment/tissue enrichment)
 *   4. Master regulator identification (hypergeometric enrichment)
 */

nextflow.enable.dsl=2

// --- Parameters ---
params.data_dir        = "${projectDir}/data"
params.output_dir      = "${projectDir}/results"
params.config          = "${projectDir}/config/default.yaml"

// GRN inference
params.method          = "ensemble"   // "ensemble" or "gnn"
params.n_trees         = 1000
params.n_bootstrap     = 100
params.n_steps_lars    = 5
params.n_mc_shuffles   = 1
params.th_lead         = 0.95
params.th_support      = 0.95
params.n_grn_support   = 1
params.seed            = 1234

// TFBS
params.tfbs_method     = "pwm"        // "pwm" or "cnn"
params.th_pre_tss      = 1000
params.th_post_tss     = 200
params.p_cutoff        = 0.05

// Context annotation
params.th_diffexp      = 0.05
params.th_pval_treat   = 0.05
params.th_pval_tissue  = 0.05

// Resources
params.cpus            = 4


// --- Processes ---

process GRN_INFERENCE {
    cpus params.cpus
    memory '16 GB'
    publishDir "${params.output_dir}/grn", mode: 'copy'

    input:
    path data_dir

    output:
    path "grn_lead_support.npy", emit: grn
    path "grn_rf.npy"
    path "grn_lr.npy"
    path "grn_clr.npy"

    script:
    """
    python3 -c "
import numpy as np
from merit.utils.data import load_datasets
from merit.inference import infer_grn_ensemble

data = load_datasets(
    genes_file='${data_dir}/gene_expression/genes.txt',
    experiment_ids_file='${data_dir}/gene_expression/ids_differentialExpressionSamples.txt',
    fold_change_file='${data_dir}/gene_expression/m.foldChange_differentialExpression.txt',
    pvalue_file='${data_dir}/gene_expression/m.pvalue_differentialExpression.txt',
    experiment_annotation_file='${data_dir}/gene_expression/experiment_annotation.txt',
    tf_annotation_file='${data_dir}/motifs/df.transcriptionFactorAnnotation.txt',
)

result = infer_grn_ensemble(
    expression=data.fold_change,
    genes=data.genes,
    regulators=data.regulators,
    n_trees=${params.n_trees},
    n_bootstrap=${params.n_bootstrap},
    n_steps_lars=${params.n_steps_lars},
    n_mc_shuffles=${params.n_mc_shuffles},
    th_lead=${params.th_lead},
    th_support=${params.th_support},
    n_grn_support=${params.n_grn_support},
    seed=${params.seed},
    n_cpus=${params.cpus},
)

np.save('grn_lead_support.npy', result['lead_support'])
np.save('grn_rf.npy', result['rf'])
np.save('grn_lr.npy', result['lr'])
np.save('grn_clr.npy', result['clr'])
    "
    """
}


process TFBS_SCAN {
    cpus params.cpus
    memory '8 GB'
    publishDir "${params.output_dir}/tfbs", mode: 'copy'

    input:
    path data_dir

    output:
    path "tfbs_matrix.npy", emit: tfbs
    path "tfbs_tfs.txt"
    path "tfbs_genes.txt"

    script:
    """
    python3 -c "
import numpy as np
from merit.tfbs import scan_tfbs

tfs, genes, mat = scan_tfbs(
    tf_motif_file='${data_dir}/motifs/TF_to_Motif_IDs.txt',
    pwm_file='${data_dir}/motifs/Transcription_factor_weight_matrix_Arabidopsis_thaliana.txt',
    promoter_file='${data_dir}/motifs/TAIR10_upstream_1000_20101104.txt',
    gene_seq_file='${data_dir}/motifs/TAIR10_seq_20110103_representative_gene_model_updated.txt',
    th_pre_tss=${params.th_pre_tss},
    th_post_tss=${params.th_post_tss},
    p_cutoff=${params.p_cutoff},
)

np.save('tfbs_matrix.npy', mat)
with open('tfbs_tfs.txt', 'w') as f:
    f.write('\\n'.join(tfs))
with open('tfbs_genes.txt', 'w') as f:
    f.write('\\n'.join(genes))
    "
    """
}


process COMBINE_GRN_TFBS {
    memory '8 GB'
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path grn
    path tfbs
    path data_dir

    output:
    path "blueprint_grn.csv", emit: blueprint

    script:
    """
    python3 -c "
import numpy as np
import pandas as pd

grn = np.load('${grn}')
tfbs = np.load('${tfbs}')

# Load gene/TF names
genes = open('${data_dir}/gene_expression/genes.txt').read().strip().split('\\n')
tf_ann = pd.read_csv('${data_dir}/motifs/df.transcriptionFactorAnnotation.txt', sep='\\t')
regulators = sorted(tf_ann.loc[tf_ann['TF_ID'].isin(genes), 'TF_ID'].unique())

# Intersect dimensions and element-wise multiply
# GRN is (n_regulators x n_genes), TFBS is (n_tfs_with_motifs x n_promoter_genes)
# Need to align by shared TFs and genes
# For now, output GRN filtered by TFBS where available
blueprint = grn  # Placeholder — proper alignment needed

# Convert to edge list
edges = []
for i in range(blueprint.shape[0]):
    for j in range(blueprint.shape[1]):
        if blueprint[i, j] > 0:
            edges.append({'TF': regulators[i] if i < len(regulators) else f'TF_{i}',
                          'TG': genes[j] if j < len(genes) else f'Gene_{j}'})

df = pd.DataFrame(edges)
df.to_csv('blueprint_grn.csv', index=False)
print(f'Blueprint GRN: {len(df)} edges')
    "
    """
}


process CONTEXT_ANNOTATION {
    cpus params.cpus
    memory '8 GB'
    publishDir "${params.output_dir}/context_specific_networks", mode: 'copy'

    input:
    path blueprint
    path data_dir

    output:
    path "*.csv", emit: subnetworks

    script:
    """
    python3 -c "
import pandas as pd

blueprint = pd.read_csv('${blueprint}')
# Context annotation produces individual CSV files per treatment-tissue pair
# Full implementation uses hypergeometric test on co-differential expression
print(f'Context annotation: {len(blueprint)} blueprint edges')
# Placeholder — write at least one output
blueprint.to_csv('all_contexts.csv', index=False)
    "
    """
}


process MASTER_REGULATORS {
    memory '4 GB'
    publishDir "${params.output_dir}/master_regulators", mode: 'copy'

    input:
    path subnetworks
    path data_dir

    output:
    path "bottom_level_master_regulators.csv"
    path "top_level_master_regulators.csv"

    script:
    """
    python3 -c "
import pandas as pd
# Placeholder for enrichment analysis
pd.DataFrame(columns=['tf','domain','fc','pval','context']).to_csv('bottom_level_master_regulators.csv', index=False)
pd.DataFrame(columns=['tf','n_tgs','fc','pval','context']).to_csv('top_level_master_regulators.csv', index=False)
    "
    """
}


// --- Workflow ---

workflow {
    data_ch = Channel.fromPath(params.data_dir, type: 'dir')

    // Steps 1 and 2 run in parallel (independent)
    GRN_INFERENCE(data_ch)
    TFBS_SCAN(data_ch)

    // Step 2.5: Combine GRN with TFBS filter
    COMBINE_GRN_TFBS(GRN_INFERENCE.out.grn, TFBS_SCAN.out.tfbs, data_ch)

    // Step 3: Context-specific annotation
    CONTEXT_ANNOTATION(COMBINE_GRN_TFBS.out.blueprint, data_ch)

    // Step 4: Master regulators
    MASTER_REGULATORS(CONTEXT_ANNOTATION.out.subnetworks, data_ch)
}
