#' This function looks for replicates in the omics data frame and replaces replicates with sample mean 
#'
#' @param omics_df is a data frame whose column names are TCGA sample barcode
#'
#' @return
#' @export
#'
#' @examples
#' dataFilt_hugo <- find_replicates(dataFilt_hugo)
find_replicates <- function(omics_df){
  # Delete normal samples and non-primary tumor samples
  normal_index <- which(as.numeric(str_sub(colnames(omics_df), 14, 15)) > 9)
  omics_df <- omics_df[, -normal_index]
  nonTP_index <- which(as.numeric(str_sub(colnames(omics_df), 14, 15)) > 1)
  omics_df <- omics_df[, -nonTP_index]
  
  # Find total patients
  total_patients <- str_sub(colnames(omics_df), 1, 12)
  
  # Find replicated sample codes
  rep_patients <- unique(total_patients[duplicated(total_patients)])
  if (length(rep_patients) != 0){
    for (p in rep_patients){
      tmp <- omics_df %>% select(starts_with(p))
      
      # Calculate mean of the replicated samples
      row_means <- apply(tmp, 1, mean, na.rm=TRUE)
      row_means <- round(row_means, 3)
      omics_df <- cbind(omics_df, row_means)
      
      # Rename last column
      colnames(omics_df)[length(colnames(omics_df))] <- p
      # Remain integers
      omics_df[,length(colnames(omics_df))] <- round(omics_df[,length(colnames(omics_df))])
      # Delete replicates
      omics_df <- omics_df[, !colnames(omics_df) %in% colnames(tmp)] 
    }
  }
  # Change all column names to sample ID
  colnames(omics_df) <- str_sub(colnames(omics_df), 1, 12)
  return(omics_df)
}




#' This function wrangles the MAF file to indicate the mutation type information of the genes for each sample
#'
#' @param mu_df should be a data frame of the mutation data
#' @param patientIDs is a vector of samples for which you want to build the mutation table
#'
#' @return a large list of data frames
#' @export
#'
#' @examples
#' mutation_list <- Mu_wrangle(mu_df = BRCA_mu, samples = common_samples, cscape = FALSE)
Mu_wrangle <- function(mu_df, patientIDs, cscape = TRUE){
  
  # Preprocessing
  mu_df <- mu_df %>% 
    # delete non-primary tumor samples
    filter(as.numeric(str_sub(Tumor_Sample_Barcode, 14, 15)) == 1) %>% 
    # add sample code information
    mutate(Patients = str_sub(Tumor_Sample_Barcode, 1, 12)) %>% 
    # keep input samples
    filter(Patients %in% patientIDs)

  CDS_df <- mu_df %>% distinct(Gene, .keep_all = TRUE)
  CDS_df["CDS_length"] <- sapply(strsplit(CDS_df$CDS_position, "\\/"), `[`, 2)      # annotate CDS length of the genes
  CDS_df <- CDS_df %>% 
    select(Gene, CDS_length) %>% 
    na.omit()
  
  # Make empty list
  mutation_list <- list()
  mut_types <- c("5'Flank", "5'UTR", "Nonsense_Mutation", "Splice_Site", "Splice_Region", "Intron", "3'UTR", "3'Flank", "RNA")
  for (id in patientIDs){
    
    # Find unique mutated genes in sample
    patient_df <- mu_df %>% 
      filter(Patients == id)
    
    # Extract unique features
    genes <- unique(patient_df$Gene)
    variant_class <- unique(patient_df$Variant_Classification)
    
    # Generate empty mutation matrix with only variant classification
    mutation <- matrix(data = 0,
                       nrow = length(genes), 
                       ncol = 9,
                       dimnames = list(genes, mut_types)) # according to consequence table
    # Add matrix values
    for (gene in genes){
      
      # Count driving mutations based on CScape score for each gene
      if (cscape == TRUE){
        driving_mu <- patient_df %>% 
          filter(Gene == gene) %>% 
          filter(CScape_Mut_Class == "Driver")
        
        if (nrow(driving_mu) != 0){
          # Assign 1 to certain mutation type with driving mutations
          mutation_name <- driving_mu$Variant_Classification
          for (name in mutation_name){
            if (name %in% colnames(mutation)){
              mutation[gene, name] <- mutation[gene, name] + 1
            }
          }
        }
      }
      # Count all mutation numbers
      else if (cscape == FALSE){
        driving_mu <- patient_df %>% 
          filter(Gene == gene)
        mutation_name <- driving_mu$Variant_Classification
        for (name in mutation_name){
          if (name %in% colnames(mutation)){
            mutation[gene, name] <- mutation[gene, name] + 1
          }
        }
      }
      else {
        print("Please assign a logical value to cscape variable.")
      }
    }
    # Convert matrix and add to list
    mutation_list[[id]] <- tibble::rownames_to_column(as.data.frame(mutation), "Gene")
  }

  # Change column names in mutation list of data frames
  for (i in 1:length(names(mutation_list))){
    colnames(mutation_list[[i]])[-1] <- paste0(colnames(mutation_list[[i]][,-1]),
                                                    "_",
                                                    names(mutation_list)[i])
  }

  # Pivot mutation data wider
  output_df <- mutation_list %>%
    purrr::reduce(full_join, by = "Gene") %>%
    replace(is.na(.), 0) %>%
    inner_join(CDS_df, by = "Gene") %>%
    # Reduce the effect of CDS length (increase the scale of counts to the same as CDS length)
    mutate_at(vars(-Gene), list(~ round((as.numeric(.)*10000)/as.numeric(CDS_length))), 3) %>%
    select(-CDS_length)

  # Average across patients  
  for (mut in mut_types){
    mut_mean <- output_df %>%
      select(starts_with(mut)) %>% 
      mutate(mean = rowMeans(.)) %>% 
      select(mean)
    colnames(mut_mean)[1] <- mut
    output_df <-cbind(output_df, mut_mean)
  }

  # Delete patient-specific data
  genewise_mut_df <- output_df %>%
    select(Gene, all_of(mut_types))
  
  # Keep only 3 digits
  genewise_mut_df[, -1] <- round(genewise_mut_df[, -1], 3)
  return(genewise_mut_df)
}