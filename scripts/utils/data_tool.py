import pandas as pd
import numpy as np
import torch
from scipy.sparse import vstack, hstack, coo_matrix, csr_matrix

def split_ID(x):
    return x.split(".")[0]


def encode_label(df):
    if df["DElabel"] == "NonDEG":
        return 0
    elif df["DElabel"] == "DOWN":
        return 1
    else:
        return 2


def row_divide(sparse_m1, sparse_m2):
    array2 = sparse_m2.toarray()
    inverse_array2 = 1 / array2
    sparse_m1 = sparse_m1.transpose()
    sparse_m2 = csr_matrix(np.diag(inverse_array2.flatten()))
    return (sparse_m1 * sparse_m2).transpose()


def load_ml_data(merged_tcga_file,
                 mRNA_data_loc,
                 promoter_data_loc,
                 cell_line):
    
    # merged_tcga_data: a full path to the merged TCGA data
    # mRNA_data_loc: a full path to a directory where mRNA annotation data is stored
    # promoter_data_loc:  a full path to a directory where promoter annotation data is stored
    # cell_line: name of the cell line in samller case          

    # TCGA data loading
    merged_tcga_data = pd.read_csv(merged_tcga_file)
    merged_tcga_data.drop_duplicates(subset=['Gene'], keep=False)

    # Change target variable
    merged_tcga_data['DEclass'] = merged_tcga_data.apply(encode_label, axis=1)

    # Delete unnecessary columns
    merged_tcga_data = merged_tcga_data.drop(columns = ['DElabel'], axis = 1)

    # mRNA data preparation
    mRNA_data = pd.read_pickle(mRNA_data_loc+'encode_'+cell_line+'_rna.pkl')
    mRNA_data['Name'] = mRNA_data['Name'].apply(split_ID)
    # print(mRNA_data.iloc[0,1])
    # print("-------------------------------")

    # Extract intersected genes
    mRNA_data = pd.merge(merged_tcga_data[['Gene']], mRNA_data, how='inner', left_on='Gene', right_on='Name')
    mRNA_data = mRNA_data.drop('Gene', axis=1)
    # mRNA_data.Name = mRNA_data.Name.reset_index(drop=True)
    # print(mRNA_data.iloc[0,1])
    # print("-------------------------------")


    # Load feature names
    mRNA_feature_name = pd.read_csv(mRNA_data_loc+'encode_'+cell_line+'_rna_feature_name.txt.gz', 
                                    header=None,compression='gzip')
    mRNA_feature_name = mRNA_feature_name.values
    # print("mRNA_feature_name\n", mRNA_feature_name)
    # Reset the index of gene names

    # Convert to dense matrix
    for i in range(mRNA_data.shape[0]):
        dense_matrix = mRNA_data.iloc[i, 1].todense()
        mRNA_data.at[i, 'feature_vec'] = dense_matrix
    
    # Filter features
    has_annot = list(map(lambda x: np.sum(x, axis=1), mRNA_data.iloc[:,1])) # rowwise sum-up of the counts for each gene
    # has_annot = list(map(lambda x: np.sum(x, axis=1), mRNA_data['feature_vec']))
    has_annot = np.hstack(has_annot)  # horizontal concatination
    indx = np.where(np.sum(has_annot>0, axis=1) >= 30)[0] # detect features binding to genes greater than 30
    for i in range(mRNA_data.shape[0]):
        dense_matrix = mRNA_data.at[i, 'feature_vec']  # Ensure it's already converted to dense
        filtered_matrix = dense_matrix[indx, :]  # Apply row filtering
        mRNA_data.at[i, 'feature_vec'] = filtered_matrix  # Update the DataFrame with the filtered matrix# delete unqualified features in the original data
    mRNA_feature_name = mRNA_feature_name[indx] # delete unqualified features in feature names


    # Promoter data preparation
    promoter_data = pd.read_pickle(promoter_data_loc+'encode_'+cell_line+'_promoter.pkl')
    promoter_data['Name'] = promoter_data['Name'].apply(split_ID)

    # Extract intersected genes
    promoter_data = pd.merge(merged_tcga_data[['Gene']], promoter_data, 
                         how='inner', left_on='Gene', right_on='Name')
    promoter_data = promoter_data.drop('Gene', axis=1)

    # Load promoter features
    promoter_feature_name = pd.read_csv(promoter_data_loc+'encode_'+cell_line+'_promoter_feature_name.txt.gz',
                                          header=None,compression='gzip')
    promoter_feature_name = promoter_feature_name.values
    
    # Convert to dense matrix
    for i in range(promoter_data.shape[0]):
        dense_matrix = promoter_data.iloc[i, 1].todense()
        promoter_data.at[i, 'feature_vec'] = dense_matrix

    # Filter features
    has_annot = list(map(lambda x: np.sum(x, axis=1), promoter_data.iloc[:,1]))
    has_annot = np.hstack(has_annot)
    indx = np.where(np.sum(has_annot>0, axis=1) >= 30)[0]

    for i in range(promoter_data.shape[0]):
        dense_matrix = promoter_data.at[i, 'feature_vec']  # Ensure it's already converted to dense
        filtered_matrix = dense_matrix[indx, :]  # Apply row filtering
        promoter_data.at[i, 'feature_vec'] = filtered_matrix
    promoter_feature_name = promoter_feature_name[indx] 
    return mRNA_data, promoter_data, mRNA_feature_name, promoter_feature_name, merged_tcga_data


def prep_ml_data_split(merged_tcga_file,
                       mRNA_data_loc,
                       promoter_data_loc,
                       cell_line,
                       train_file,
                       val_file,
                       test_file,
                       outloc,
                       shuffle="None"):
    
    # merged_tcga_loc: a full path to the merged TCGA data
    # mRNA_data_loc: a full path to a directory where mRNA annotation data is stored
    # promoter_data_loc:  a full path to a directory where promoter annotation data is stored
    # train_file: a full path to a file contating gene ids for training
    # val_file: a full path to a file contating gene ids for validation
    # test_file: a full path to a file contating gene ids for testing
    # outloc: a full path where scaling factors are saved
    # shuffle: which features are shuffled or not
    
    mRNA_data, promoter_data, mRNA_feature_name, promoter_feature_name, merged_tcga_data = load_ml_data(merged_tcga_file, mRNA_data_loc, promoter_data_loc, cell_line)
    
    
    # Split data into training, validating, and testing subsets
    test = pd.read_csv(test_file,sep="\t",header=0).values[:,0]
    val = pd.read_csv(val_file,sep="\t",header=0).values[:,0]
    train = pd.read_csv(train_file,sep="\t",header=0).values[:,0]

    # Split mRNA feature
    X_mRNA_train=mRNA_data.query("Name in @train").copy()
    X_mRNA_train.drop_duplicates(subset=['Name'], inplace=True)
    X_mRNA_val=mRNA_data.query("Name in @val").copy()
    X_mRNA_val.drop_duplicates(subset=['Name'], inplace=True)
    X_mRNA_test=mRNA_data.query("Name in @test").copy()
    X_mRNA_test.drop_duplicates(subset=['Name'], inplace=True)

    # Split promoter feature
    X_promoter_train=promoter_data.query("Name in @train").copy()
    X_promoter_train.drop_duplicates(subset=['Name'], inplace=True)
    X_promoter_val=promoter_data.query("Name in @val").copy()
    X_promoter_val.drop_duplicates(subset=['Name'], inplace=True)
    X_promoter_test=promoter_data.query("Name in @test").copy()
    X_promoter_test.drop_duplicates(subset=['Name'], inplace=True)


    # Split target data
    Y_train=merged_tcga_data.query("Gene in @train").copy().DEclass
    Y_val=merged_tcga_data.query("Gene in @val").copy().DEclass
    Y_test=merged_tcga_data.query("Gene in @test").copy().DEclass

    
    Y_feature_norm_stats = pd.DataFrame({'feature_name' : ["nonDEG", "downDEG", "upDEG"],
                                        'row_indx': range(3),
                                        'feature_type' : 'deg_stat'})

    omics_feature_norm_stats = pd.DataFrame({'feature_name' : merged_tcga_data.columns.tolist()[1:-1],
                                            'row_indx': range(len(merged_tcga_data.columns.tolist()[1:-1])),
                                            'feature_type' : 'omics_range'})

    # Normalize mRNA features
    # Horizontally splice the sparse matrix of all genes
    mRNA_std = hstack(X_mRNA_train.values[:,1]).max(axis=1).tocsr()

    # std = hstack([csr_matrix(item) for item in X_mRNA_train.values[:,1]]).max(axis=1).tocsr()

    # Special treatment for STD=0
    # mRNA_std[mRNA_std == 0] = 1
    mRNA_std = mRNA_std + (mRNA_std == 0).multiply(1)

    for i in range(len(X_mRNA_train.values[:,1])):
        X_mRNA_train.values[i,1] = row_divide(X_mRNA_train.values[i,1], mRNA_std)
    for i in range(len(X_mRNA_val.values[:,1])):
        X_mRNA_val.values[i,1] = row_divide(X_mRNA_val.values[i,1], mRNA_std)
    for i in range(len(X_mRNA_test.values[:,1])):
        X_mRNA_test.values[i,1] = row_divide(X_mRNA_test.values[i,1], mRNA_std)

    # Store normalization stats 
    mRNA_feature_norm_stats = pd.DataFrame({'feature_name' : mRNA_feature_name.flatten(),
                                            'row_indx': range(len(mRNA_feature_name)),
                                            'feature_type' : 'mRNA_range',
                                            'max':mRNA_std.toarray().flatten()})

    # Normalizing promoter features
    promoter_std = hstack(X_promoter_train.values[:,1]).max(axis=1).tocsr()
    # std = hstack([csr_matrix(item) for item in X_promoter_train.values[:,1]]).max(axis=1).tocsr()

    # Special treatment for STD=0
    # promoter_std[promoter_std == 0] = 1
    promoter_std = promoter_std + (promoter_std == 0).multiply(1)

    # Scaling
    for i in range(len(X_promoter_train.values[:,1])):
        X_promoter_train.values[i,1] = row_divide(X_promoter_train.values[i,1], promoter_std)
    for i in range(len(X_promoter_val.values[:,1])):
        X_promoter_val.values[i,1] = row_divide(X_promoter_val.values[i,1], promoter_std)
    for i in range(len(X_promoter_test.values[:,1])):
        X_promoter_test.values[i,1] = row_divide(X_promoter_test.values[i,1], promoter_std)


    # Store normalization stats 
    promoter_feature_norm_stats = pd.DataFrame({'feature_name' : promoter_feature_name.flatten(),
                                                 'row_indx': range(len(promoter_feature_name)),
                                                'feature_type' : 'promoter_range',
                                                'max':promoter_std.toarray().flatten()})

    # conbine all stats 
    feature_norm_stats=pd.concat((Y_feature_norm_stats, omics_feature_norm_stats, mRNA_feature_norm_stats, promoter_feature_norm_stats),sort=False)  

    feature_norm_stats.to_csv(outloc+'feature_norm_stats.txt',sep=",")
    
    # Shuffle features
    if shuffle=="all":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_train.shape[0]))
        X_mRNA_train['feature_vec'][i] =X_mRNA_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_val.shape[0]))
        X_mRNA_val['feature_vec'][i] =X_mRNA_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_test.shape[0]))
        X_mRNA_test['feature_vec'][i] =X_mRNA_test.iloc[shuffle_indices,1]

        shuffle_indices = np.random.permutation(np.arange(X_promoter_train.shape[0]))
        X_promoter_train['feature_vec'][i] =X_promoter_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_val.shape[0]))
        X_promoter_val['feature_vec'][i] =X_promoter_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_test.shape[0]))
        X_promoter_test['feature_vec'][i] =X_promoter_test.iloc[shuffle_indices,1]
    elif shuffle=="DNA":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_promoter_train.shape[0]))
        X_promoter_train['feature_vec'][i] =X_promoter_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_val.shape[0]))
        X_promoter_val['feature_vec'][i] =X_promoter_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_test.shape[0]))
        X_promoter_test['feature_vec'][i] =X_promoter_test.iloc[shuffle_indices,1]
    elif shuffle=="RNA":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_train.shape[0]))
        X_mRNA_train['feature_vec'][i] =X_mRNA_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_val.shape[0]))
        X_mRNA_val['feature_vec'][i] =X_mRNA_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_test.shape[0]))
        X_mRNA_test['feature_vec'][i] =X_mRNA_test.iloc[shuffle_indices,1]

    return Y_train, Y_val, Y_test, X_mRNA_train, X_mRNA_val, X_mRNA_test, X_promoter_train, X_promoter_val, X_promoter_test



def batch_iter(tcga_train,
               X_mRNA_train,
               X_promoter_train,
               Y_train,
               batch_size,
               shuffle=True):
    
    # tcga_train: tcga data
    # X_mRNA_train: mRNA features
    # X_promoter_train: promoter features
    # Y_train: target data
    # batch_size: The number of data in each batch
    # shuffle=True: Shuffling the order of data in each epoch
    
    # Number of data in each batch
    data_size = len(Y_train)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    
    # Obtain size of mRNA feature matrix
    n_feature_mRNA=X_mRNA_train[0].shape[0]
    
    # Obtain size of promoter feature matrix
    n_feature_promoter=X_promoter_train[0].shape[0]
    
    def data_generator():
        while True:
            
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_tcga_data = tcga_train.iloc[shuffle_indices]
                shuffled_data_mRNA = X_mRNA_train[shuffle_indices]
                shuffled_data_promoter = X_promoter_train[shuffle_indices]
                shuffled_labels = Y_train[shuffle_indices]
            else:
                shuffled_tcga_data = tcga_train
                shuffled_data_mRNA = X_mRNA_train
                shuffled_data_promoter = X_promoter_train
                shuffled_labels = Y_train
                
            # Generate data for each batch
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                # Prepare mRNA feature matrix
                # Obtain max length of mRNA sequence in this batch
                seq_len_mRNA = list(map(lambda x: np.shape(x)[1], shuffled_data_mRNA[start_index: end_index]))
                max_seq_mRNA = max(seq_len_mRNA)  # Convert map object to list and then find max

                # Initialize mRNA feature matrix
                X_mRNA = np.zeros((end_index-start_index, max_seq_mRNA, n_feature_mRNA))
                k = 0
                for i in range(start_index,end_index):
                    X_mRNA[k,0:seq_len_mRNA[k],:] = shuffled_data_mRNA[i].transpose()
                    k = k+1
                X_mRNA = torch.from_numpy(X_mRNA).float()
                    
                # Prepare promoter feature matrix
                # Obtain max length of promoter sequence in this batch
                # seq_len_promoter=map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index])
                seq_len_promoter = list(map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index]))
                max_seq_promoter = np.max(seq_len_promoter)
                
                # Initialize promoter feature matrix
                X_promoter = np.zeros((end_index-start_index, max_seq_promoter, n_feature_promoter))
                k = 0
                for i in range(start_index,end_index):
                    X_promoter[k,0:seq_len_promoter[k],:]=shuffled_data_promoter[i].transpose()
                    k = k+1
                X_promoter = torch.from_numpy(X_promoter).float()
                
                # Prepare tcga batch data
                X_tcga = shuffled_tcga_data.values[start_index:end_index]
                try:
                    X_tcga = torch.from_numpy(X_tcga).float()
                except:
                    X_tcga = X_tcga

                # Prepare target data
                Y = shuffled_labels[start_index: end_index]
                Y = torch.from_numpy(Y).long()
                
                yield [X_mRNA,X_promoter], X_tcga, Y

    return num_batches_per_epoch, data_generator()





def batch_iter_GradSHAP(tcga_train,
                        X_mRNA_train,
                        X_promoter_train,
                        Y_train,
                        batch_size, 
                        med_mRNA_len,
                        med_promoter_len,
                        shuffle=True):
    
    # Number of data in each batch
    data_size = len(Y_train)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    
    # Obtain the size of mRNA feature matrix
    n_feature_mRNA=X_mRNA_train[0].shape[0]
    
    # Obtain the size of promotor feature matrix
    n_feature_promoter=X_promoter_train[0].shape[0]
    
    def data_generator():
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_tcga_data = tcga_train.iloc[shuffle_indices]
                shuffled_data_mRNA = X_mRNA_train[shuffle_indices]
                shuffled_data_promoter = X_promoter_train[shuffle_indices]
                shuffled_labels = Y_train[shuffle_indices]
            else:
                shuffled_tcga_data = tcga_train
                shuffled_data_mRNA = X_mRNA_train
                shuffled_data_promoter = X_promoter_train
                shuffled_labels = Y_train

            # Generate data for each batch
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                # Prepare mRNA feature matrix
                # Obtain max length of mRNA sequence in this batch
                seq_len_mRNA=list(map(lambda x: np.shape(x)[1], shuffled_data_mRNA[start_index: end_index]))
                max_seq_mRNA=np.max(seq_len_mRNA) 
                max_seq_mRNA=np.max([max_seq_mRNA,med_mRNA_len])
                
                # Initializing mRNA feature matrix (concate all data of the batch based on the maximum seq length)
                X_mRNA=np.zeros((end_index-start_index, max_seq_mRNA, n_feature_mRNA))
                k=0
                for i in range(start_index,end_index):
                    X_mRNA[k,0:seq_len_mRNA[k],:]=shuffled_data_mRNA[i].transpose()
                    k=k+1
                X_mRNA = torch.from_numpy(X_mRNA).float()
                    
                # Prepare promoter feature matrix
                # Obtain max length of promoter sequence in this batch
                seq_len_promoter=list(map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index]))
                max_seq_promoter=np.max(seq_len_promoter)
                max_seq_promoter=np.max([max_seq_promoter,med_promoter_len])
                
                # Initializing promoter feature matrix
                X_promoter=np.zeros((end_index-start_index, max_seq_promoter, n_feature_promoter))
                k=0
                for i in range(start_index,end_index):
                    X_promoter[k,0:seq_len_promoter[k],:]=shuffled_data_promoter[i].transpose()
                    k=k+1  
                X_promoter = torch.from_numpy(X_promoter).float()                 
                
                # Prepare tcga batch data
                X_tcga = shuffled_tcga_data.values[start_index:end_index]
                try:
                    X_tcga = torch.from_numpy(X_tcga).float()
                except:
                    X_tcga = X_tcga

                # Prepare target data
                Y = shuffled_labels[start_index: end_index]
                Y = torch.from_numpy(Y).long()
                
                yield [X_mRNA,X_promoter], X_tcga, Y             

    return num_batches_per_epoch, data_generator()




def length_align(test_batch, tcga_train, X_mRNA_train, X_promoter_train, Y_train, sample_size, max_test_mRNA, max_test_promoter, shuffle=True, seed=42):
    """This function aligns the mRNA and promoter length between training batch test batch"""
    # Number of data in each batch
    data_size = len(Y_train)
    
    # Obtain the size of mRNA feature matrix
    n_feature_mRNA = X_mRNA_train[0].shape[0]
    
    # Obtain the size of promotor feature matrix
    n_feature_promoter = X_promoter_train[0].shape[0]
    
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_tcga_data = tcga_train.iloc[shuffle_indices]
        shuffled_data_mRNA = X_mRNA_train[shuffle_indices]
        shuffled_data_promoter = X_promoter_train[shuffle_indices]
    else:
        shuffled_tcga_data = tcga_train
        shuffled_data_mRNA = X_mRNA_train
        shuffled_data_promoter = X_promoter_train

    # Extract random data with sample size
    np.random.seed(seed)
    indx = np.random.choice(data_size, 100, replace=False)
    shuffled_data_mRNA = shuffled_data_mRNA[indx]
    shuffled_data_promoter = shuffled_data_promoter[indx]
    shuffled_tcga_data = shuffled_tcga_data.iloc[indx]

    # Prepare mRNA feature matrix
    # Obtain max length of mRNA sequence in this batch
    seq_len_mRNA=list(map(lambda x: np.shape(x)[1], shuffled_data_mRNA))
    max_train_mRNA=np.max(seq_len_mRNA) 
    if max_train_mRNA <= max_test_mRNA:
        max_seq_mRNA = max_test_mRNA
    else:
        max_seq_mRNA = max_train_mRNA

    # Initializing mRNA feature matrix (concate all data of the batch based on the maximum seq length)
    X_mRNA = np.zeros((sample_size, max_seq_mRNA, n_feature_mRNA))
    k = 0
    for i in range(sample_size):
        X_mRNA[k,0:seq_len_mRNA[k],:] = shuffled_data_mRNA[i].transpose()
        k = k+1
    X_mRNA = torch.from_numpy(X_mRNA).float()

    # Change the dimension of the mRNA test data
    if max_train_mRNA > max_test_mRNA:
        gap_len = max_train_mRNA - max_test_mRNA
        test_batch[0][0] = np.pad(test_batch[0][0], ((0, 0), (0, gap_len), (0, 0)), mode='constant', constant_values=0)
        test_batch[0][0] = torch.from_numpy(test_batch[0][0]).float()


    # Prepare promoter feature matrix
    # Obtain max length of promoter sequence in this batch
    seq_len_promoter=list(map(lambda x: np.shape(x)[1], shuffled_data_promoter))
    max_train_promoter=np.max(seq_len_promoter) 
    if max_train_promoter <= max_test_promoter:
        max_seq_promoter = max_test_promoter
    else:
        max_seq_promoter = max_train_promoter
    
    # Initializing promoter feature matrix
    X_promoter = np.zeros((sample_size, max_seq_promoter, n_feature_promoter))
    k = 0
    for i in range(sample_size):
        X_promoter[k,0:seq_len_promoter[k],:] = shuffled_data_promoter[i].transpose()
        k=k+1  
    X_promoter = torch.from_numpy(X_promoter).float()  

    # Change the dimension of the mRNA test data
    if max_train_promoter > max_test_promoter:
        gap_len = max_train_promoter - max_test_promoter
        test_batch[0][1] = np.pad(test_batch[0][1], ((0, 0), (0, gap_len), (0, 0)), mode='constant', constant_values=0)
        test_batch[0][1] = torch.from_numpy(test_batch[0][1]).float()


    # Prepare tcga batch data
    X_tcga = shuffled_tcga_data.values
    try:
        X_tcga = torch.from_numpy(X_tcga).float()
    except:
        X_tcga = X_tcga

    train_batch = [X_mRNA,X_promoter], X_tcga

    return train_batch, test_batch








def pancan_load_ml_data(deg_data_file,
                 mRNA_data_loc,
                 promoter_data_loc):
    
    # deg_data_file: a full path to the cancer expression data
    # mRNA_data_loc: a full path to a directory where mRNA annotation data is stored
    # promoter_data_loc:  a full path to a directory where promoter annotation data is stored

    # Differential expression calculation 
    deg_data = pd.read_csv(deg_data_file,sep="\t")
    # Check the first column
    if deg_data.columns[0] != 'Name':
        print('The first column must be Name')
        sys.exit(1) 
    # Check the last column
    if deg_data.columns[-1] != "MedianExp":
        print('The last column must be MedianExp')
        sys.exit(1) 
    # Calculation 
    for i in range(deg_data.shape[0]):
        deg_data.iloc[i, 1:-1] = deg_data.values[i, 1:-1] - deg_data['MedianExp'][i]


    # mRNA data preparation
    mRNA_data = pd.read_pickle(mRNA_data_loc+"output_postar_rna.pkl")
    mRNA_data['Name'] = mRNA_data['Name'].apply(split_ID)

    # Extract intersected genes
    mRNA_data = pd.merge(deg_data[['Name']], mRNA_data, how='inner', on='Name')

    # Load feature names
    mRNA_feature_name = pd.read_csv(mRNA_data_loc+"output_postar_rna_feature_name.txt.gz", 
                                    header=None,compression='gzip')
    mRNA_feature_name = mRNA_feature_name.values

    # Merge with miRNA
    range_data_temp = pd.read_pickle(mRNA_data_loc+"output_targetscan_rna.pkl")
    range_data_temp['Name'] = range_data_temp['Name'].apply(split_ID)
    
    # Filter genes
    range_data_temp = pd.merge(deg_data[['Name']], range_data_temp, how='inner', on='Name')
    # range_data_temp = range_data_temp.drop('Gene', axis=1)
    # range_data_temp.Name = range_data_temp.Name.reset_index(drop=True)
    # print(range_data_temp.iloc[0,1])

    # Read feature names
    feature_name_temp = pd.read_csv(mRNA_data_loc+"/output_targetscan_rna_feature_name.txt.gz",
                                        header=None,compression='gzip')
    feature_name_temp = feature_name_temp.values
    # print("feature_name_temp\n", feature_name_temp)

    # Check gene order is identical
    if sum(mRNA_data.Name != range_data_temp.Name) > 0:
        raise Exception("gene name does not math!")

    # Remove exon
    indx = feature_name_temp[:,0] != "exon"
    # print("indx\n", indx)

    for i in range(mRNA_data.shape[0]):
        mRNA_data.iat[i, 1] = vstack((mRNA_data.iloc[i, 1], range_data_temp.iloc[i, 1][indx]))  # Don't use ".iloc" here otherwise it didn't work

    # Update feature names
    mRNA_feature_name = np.concatenate((mRNA_feature_name,feature_name_temp[indx]))

    # Convert to dense matrix
    for i in range(mRNA_data.shape[0]):
        dense_matrix = mRNA_data.iloc[i, 1].todense()
        mRNA_data.at[i, 'feature_vec'] = dense_matrix
    
    # Filter features
    has_annot = list(map(lambda x: np.sum(x, axis=1), mRNA_data.iloc[:,1])) # rowwise sum-up of the counts for each gene
    # has_annot = list(map(lambda x: np.sum(x, axis=1), mRNA_data['feature_vec']))
    has_annot = np.hstack(has_annot)  # horizontal concatination
    indx = np.where(np.sum(has_annot>0, axis=1) >= 30)[0] # detect features binding to genes greater than 30
    for i in range(mRNA_data.shape[0]):
        dense_matrix = mRNA_data.at[i, 'feature_vec']  # Ensure it's already converted to dense
        filtered_matrix = dense_matrix[indx, :]  # Apply row filtering
        mRNA_data.at[i, 'feature_vec'] = filtered_matrix  # Update the DataFrame with the filtered matrix# delete unqualified features in the original data
    mRNA_feature_name = mRNA_feature_name[indx] # delete unqualified features in feature names


    # Promoter data preparation
    promoter_data = pd.read_pickle(promoter_data_loc+"output_gtrd_promoter.pkl")
    promoter_data['Name'] = promoter_data['Name'].apply(split_ID)

    # Extract intersected genes
    promoter_data = pd.merge(deg_data[['Name']], promoter_data, how='inner', on='Name')
    # promoter_data = promoter_data.drop('Gene', axis=1)

    # Load promoter features
    promoter_feature_name = pd.read_csv(promoter_data_loc+"output_gtrd_promoter_feature_name.txt.gz",
                                          header=None,compression='gzip')
    promoter_feature_name = promoter_feature_name.values
    
    # Convert to dense matrix
    for i in range(promoter_data.shape[0]):
        dense_matrix = promoter_data.iloc[i, 1].todense()
        promoter_data.at[i, 'feature_vec'] = dense_matrix

    # Filter features
    has_annot = list(map(lambda x: np.sum(x, axis=1), promoter_data.iloc[:,1]))
    has_annot = np.hstack(has_annot)
    indx = np.where(np.sum(has_annot>0, axis=1) >= 30)[0]

    for i in range(promoter_data.shape[0]):
        dense_matrix = promoter_data.at[i, 'feature_vec']  # Ensure it's already converted to dense
        filtered_matrix = dense_matrix[indx, :]  # Apply row filtering
        promoter_data.at[i, 'feature_vec'] = filtered_matrix
    promoter_feature_name = promoter_feature_name[indx] 
    return mRNA_data, promoter_data, mRNA_feature_name, promoter_feature_name, deg_data


def pancan_prep_ml_data_split(deg_data_file,
                       mRNA_data_loc,
                       promoter_data_loc,
                       train_file,
                       val_file,
                       test_file,
                       outloc,
                       shuffle="None"):
    
    # merged_tcga_loc: a full path to the merged TCGA data
    # mRNA_data_loc: a full path to a directory where mRNA annotation data is stored
    # promoter_data_loc:  a full path to a directory where promoter annotation data is stored
    # train_file: a full path to a file contating gene ids for training
    # val_file: a full path to a file contating gene ids for validation
    # test_file: a full path to a file contating gene ids for testing
    # outloc: a full path where scaling factors are saved
    # shuffle: which features are shuffled or not
    
    mRNA_data, promoter_data, mRNA_feature_name, promoter_feature_name, deg_data = pancan_load_ml_data(deg_data_file, mRNA_data_loc, promoter_data_loc)
    
    
    # Split data into training, validating, and testing subsets
    test = pd.read_csv(test_file,sep="\t",header=0).values[:,0]
    val = pd.read_csv(val_file,sep="\t",header=0).values[:,0]
    train = pd.read_csv(train_file,sep="\t",header=0).values[:,0]

    # Split mRNA feature
    X_mRNA_train=mRNA_data.query("Name in @train").copy()
    X_mRNA_train.drop_duplicates(subset=['Name'], inplace=True)
    X_mRNA_val=mRNA_data.query("Name in @val").copy()
    X_mRNA_val.drop_duplicates(subset=['Name'], inplace=True)
    X_mRNA_test=mRNA_data.query("Name in @test").copy()
    X_mRNA_test.drop_duplicates(subset=['Name'], inplace=True)

    # Split promoter feature
    X_promoter_train=promoter_data.query("Name in @train").copy()
    X_promoter_train.drop_duplicates(subset=['Name'], inplace=True)
    X_promoter_val=promoter_data.query("Name in @val").copy()
    X_promoter_val.drop_duplicates(subset=['Name'], inplace=True)
    X_promoter_test=promoter_data.query("Name in @test").copy()
    X_promoter_test.drop_duplicates(subset=['Name'], inplace=True)


    # Split target data
    Y_train = deg_data.query("Name in @train").copy()
    Y_val = deg_data.query("Name in @val").copy()
    Y_test = deg_data.query("Name in @test").copy()
    
    # Scale fold changes
    std = Y_train.values[:,1:-1].std()  # 从第二列开始，不包含 'Name'
    for i in range(1, Y_train.shape[1]-1):  # 从第二列开始迭代
        Y_train.iloc[:, i] = Y_train.iloc[:, i] / std
        Y_val.iloc[:, i] = Y_val.iloc[:, i] / std
        Y_test.iloc[:, i] = Y_test.iloc[:, i] / std

    # Store normalization stats 
    DEG_scale_stats = pd.DataFrame({'feature_name' : np.array(Y_train.columns)[1:(Y_train.shape[1]-1)],
                                    'row_indx': range(0,Y_train.shape[1]-2),
                                    'feature_type' : 'deg_stat',
                                    'std': std})
    
    # Scale log2-TPM (assume it is located at the last column)
    std = Y_train.iloc[:, -1].std()
    Y_train.iloc[:, -1] = Y_train.iloc[:, -1] / std
    Y_val.iloc[:, -1] = Y_val.iloc[:, -1] / std
    Y_test.iloc[:, -1] = Y_test.iloc[:, -1] / std

    # Store normalization stats 
    DEG_scale_stats = pd.concat((DEG_scale_stats,
                                pd.DataFrame({'feature_name' : np.array(Y_train.columns)[(Y_train.shape[1]-1):],
                                             'row_indx': Y_train.shape[1]-2,
                                             'feature_type' : 'deg_stat',
                                             'std': std}))) 

    # Normalize mRNA features
    # Horizontally splice the sparse matrix of all genes
    mRNA_std = hstack(X_mRNA_train.values[:,1]).max(axis=1).tocsr()

    # std = hstack([csr_matrix(item) for item in X_mRNA_train.values[:,1]]).max(axis=1).tocsr()

    # Special treatment for STD=0
    # mRNA_std[mRNA_std == 0] = 1
    mRNA_std = mRNA_std + (mRNA_std == 0).multiply(1)

    for i in range(len(X_mRNA_train.values[:,1])):
        X_mRNA_train.values[i,1] = row_divide(X_mRNA_train.values[i,1], mRNA_std)
    for i in range(len(X_mRNA_val.values[:,1])):
        X_mRNA_val.values[i,1] = row_divide(X_mRNA_val.values[i,1], mRNA_std)
    for i in range(len(X_mRNA_test.values[:,1])):
        X_mRNA_test.values[i,1] = row_divide(X_mRNA_test.values[i,1], mRNA_std)

    # Store normalization stats 
    mRNA_feature_norm_stats = pd.DataFrame({'feature_name' : mRNA_feature_name.flatten(),
                                        'row_indx': range(len(mRNA_feature_name)),
                                        'feature_type' : 'mRNA_range',
                                        'max':mRNA_std.toarray().flatten()})

    # Normalizing promoter features
    promoter_std = hstack(X_promoter_train.values[:,1]).max(axis=1).tocsr()
    # std = hstack([csr_matrix(item) for item in X_promoter_train.values[:,1]]).max(axis=1).tocsr()

    # Special treatment for STD=0
    # promoter_std[promoter_std == 0] = 1
    promoter_std = promoter_std + (promoter_std == 0).multiply(1)

    # Scaling
    for i in range(len(X_promoter_train.values[:,1])):
        X_promoter_train.values[i,1] = row_divide(X_promoter_train.values[i,1], promoter_std)
    for i in range(len(X_promoter_val.values[:,1])):
        X_promoter_val.values[i,1] = row_divide(X_promoter_val.values[i,1], promoter_std)
    for i in range(len(X_promoter_test.values[:,1])):
        X_promoter_test.values[i,1] = row_divide(X_promoter_test.values[i,1], promoter_std)


    # Store normalization stats 
    promoter_feature_norm_stats = pd.DataFrame({'feature_name' : promoter_feature_name.flatten(),
                                                 'row_indx': range(len(promoter_feature_name)),
                                                'feature_type' : 'promoter_range',
                                                'max':promoter_std.toarray().flatten()})

    # conbine all stats 
    feature_norm_stats=pd.concat((DEG_scale_stats, mRNA_feature_norm_stats, promoter_feature_norm_stats), sort=False)  
    feature_norm_stats.to_csv(outloc+'feature_norm_stats.txt',sep=",")
    
    # Shuffle features
    if shuffle=="all":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_train.shape[0]))
        X_mRNA_train['feature_vec'][i] =X_mRNA_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_val.shape[0]))
        X_mRNA_val['feature_vec'][i] =X_mRNA_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_test.shape[0]))
        X_mRNA_test['feature_vec'][i] =X_mRNA_test.iloc[shuffle_indices,1]

        shuffle_indices = np.random.permutation(np.arange(X_promoter_train.shape[0]))
        X_promoter_train['feature_vec'][i] =X_promoter_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_val.shape[0]))
        X_promoter_val['feature_vec'][i] =X_promoter_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_test.shape[0]))
        X_promoter_test['feature_vec'][i] =X_promoter_test.iloc[shuffle_indices,1]
    elif shuffle=="DNA":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_promoter_train.shape[0]))
        X_promoter_train['feature_vec'][i] =X_promoter_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_val.shape[0]))
        X_promoter_val['feature_vec'][i] =X_promoter_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_promoter_test.shape[0]))
        X_promoter_test['feature_vec'][i] =X_promoter_test.iloc[shuffle_indices,1]
    elif shuffle=="RNA":
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_train.shape[0]))
        X_mRNA_train['feature_vec'][i] =X_mRNA_train.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_val.shape[0]))
        X_mRNA_val['feature_vec'][i] =X_mRNA_val.iloc[shuffle_indices,1]
        shuffle_indices = np.random.permutation(np.arange(X_mRNA_test.shape[0]))
        X_mRNA_test['feature_vec'][i] =X_mRNA_test.iloc[shuffle_indices,1]

    return Y_train, Y_val, Y_test, X_mRNA_train, X_mRNA_val, X_mRNA_test, X_promoter_train, X_promoter_val, X_promoter_test





def pancan_batch_iter(X_mRNA_train,
               X_promoter_train,
               Y_train,
               batch_size,
               shuffle=True):
    
    # X_mRNA_train: mRNA features
    # X_promoter_train: promoter features
    # Y_train: target data
    # batch_size: The number of data in each batch
    # shuffle=True: Shuffling the order of data in each epoch
    
    # Number of data in each batch
    data_size = len(Y_train)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    
    # Obtain size of mRNA feature matrix
    n_feature_mRNA=X_mRNA_train[0].shape[0]
    
    # Obtain size of promoter feature matrix
    n_feature_promoter=X_promoter_train[0].shape[0]
    
    def data_generator():
        while True:
            
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data_mRNA = X_mRNA_train[shuffle_indices]
                shuffled_data_promoter = X_promoter_train[shuffle_indices]
                shuffled_labels = Y_train[shuffle_indices]
            else:
                shuffled_data_mRNA = X_mRNA_train
                shuffled_data_promoter = X_promoter_train
                shuffled_labels = Y_train
                
            # Generate data for each batch
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                # Prepare mRNA feature matrix
                # Obtain max length of mRNA sequence in this batch
                seq_len_mRNA = list(map(lambda x: np.shape(x)[1], shuffled_data_mRNA[start_index: end_index]))
                max_seq_mRNA = max(seq_len_mRNA)  # Convert map object to list and then find max

                # Initialize mRNA feature matrix
                X_mRNA = np.zeros((end_index-start_index, max_seq_mRNA, n_feature_mRNA))
                k = 0
                for i in range(start_index,end_index):
                    X_mRNA[k,0:seq_len_mRNA[k],:] = shuffled_data_mRNA[i].transpose()
                    k = k+1
                X_mRNA = torch.from_numpy(X_mRNA).float()
                    
                # Prepare promoter feature matrix
                # Obtain max length of promoter sequence in this batch
                # seq_len_promoter=map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index])
                seq_len_promoter = list(map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index]))
                max_seq_promoter = np.max(seq_len_promoter)
                
                # Initialize promoter feature matrix
                X_promoter = np.zeros((end_index-start_index, max_seq_promoter, n_feature_promoter))
                k = 0
                for i in range(start_index,end_index):
                    X_promoter[k,0:seq_len_promoter[k],:]=shuffled_data_promoter[i].transpose()
                    k = k+1
                X_promoter = torch.from_numpy(X_promoter).float()

                # Prepare target data
                Y = shuffled_labels[start_index: end_index]
                Y = torch.from_numpy(Y).float()
                
                yield [X_mRNA,X_promoter], Y

    return num_batches_per_epoch, data_generator()





def pancan_batch_iter_GradSHAP(X_mRNA_train,
                        X_promoter_train,
                        Y_train,
                        batch_size, 
                        med_mRNA_len,
                        med_promoter_len,
                        shuffle=True):
    
    # Number of data in each batch
    data_size = len(Y_train)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    
    # Obtain the size of mRNA feature matrix
    n_feature_mRNA=X_mRNA_train[0].shape[0]
    
    # Obtain the size of promotor feature matrix
    n_feature_promoter=X_promoter_train[0].shape[0]
    
    def data_generator():
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data_mRNA = X_mRNA_train[shuffle_indices]
                shuffled_data_promoter = X_promoter_train[shuffle_indices]
                shuffled_labels = Y_train[shuffle_indices]
            else:
                shuffled_data_mRNA = X_mRNA_train
                shuffled_data_promoter = X_promoter_train
                shuffled_labels = Y_train

            # Generate data for each batch
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                # Prepare mRNA feature matrix
                # Obtain max length of mRNA sequence in this batch
                seq_len_mRNA=list(map(lambda x: np.shape(x)[1], shuffled_data_mRNA[start_index: end_index]))
                max_seq_mRNA=np.max(seq_len_mRNA) 
                max_seq_mRNA=np.max([max_seq_mRNA,med_mRNA_len])
                
                # Initializing mRNA feature matrix (concate all data of the batch based on the maximum seq length)
                X_mRNA=np.zeros((end_index-start_index, max_seq_mRNA, n_feature_mRNA))
                k=0
                for i in range(start_index,end_index):
                    X_mRNA[k,0:seq_len_mRNA[k],:]=shuffled_data_mRNA[i].transpose()
                    k=k+1
                X_mRNA = torch.from_numpy(X_mRNA).float()
                    
                # Prepare promoter feature matrix
                # Obtain max length of promoter sequence in this batch
                seq_len_promoter=list(map(lambda x: np.shape(x)[1], shuffled_data_promoter[start_index: end_index]))
                max_seq_promoter=np.max(seq_len_promoter)
                max_seq_promoter=np.max([max_seq_promoter,med_promoter_len])
                
                # Initializing promoter feature matrix
                X_promoter=np.zeros((end_index-start_index, max_seq_promoter, n_feature_promoter))
                k=0
                for i in range(start_index,end_index):
                    X_promoter[k,0:seq_len_promoter[k],:]=shuffled_data_promoter[i].transpose()
                    k=k+1  
                X_promoter = torch.from_numpy(X_promoter).float()                             
                
                # Prepare target data
                Y = shuffled_labels[start_index: end_index]
                Y = torch.from_numpy(Y).long()

                yield [X_mRNA,X_promoter], Y

    return num_batches_per_epoch, data_generator()

