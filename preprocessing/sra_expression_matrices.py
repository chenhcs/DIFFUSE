import numpy as np

def generate_full_expression_matrix(iso_id_mapping_file, expression_file, expression_matrix_output_file, isoform_train_list, isoform_test_list):
    isoform_train_list_array = np.load(isoform_train_list)
    isoform_test_list_array = np.load(isoform_test_list)
    isoform_list = np.hstack((isoform_train_list_array, isoform_test_list_array))

    # Create an isoform name(NM_) -> id map.
    isoform_name_id_map = {}
    fr = open(iso_id_mapping_file, 'r')

    header = fr.readline()
    while True:
        line = fr.readline()
        if not line:
            break
        isoform_name, gene_id, isoform_id = line.split('\t')[0: 3]
        isoform_name_id_map[isoform_name.split('.')[0]] = isoform_id
    fr.close()

    # Create an isoform id -> expression profile map.
    isoform_id_exp_map = {}
    fr = open(expression_file, 'r')
    header = fr.readline()
    while True:
        line = fr.readline()
        if not line:
            break
        line = line.split('\n')[0]
        isoform_id = line.split('\t')[0]
        expression_all = line.split('\t')[2: ]
        expression_all = np.array(expression_all)
        expression_all = expression_all.astype('float32')
        isoform_id_exp_map[isoform_id] = expression_all
    fr.close()

    expression_zeros = np.zeros(isoform_id_exp_map[isoform_id].shape[0])
    expression_zeros = expression_zeros.astype('float32')
    isoform_id_exp_map[-1] = expression_zeros

    # Generate the expression profile according to the isoform list.
    expression_matrix_all = np.zeros((isoform_list.shape[0], expression_zeros.shape[0]))
    for i in range(len(isoform_list)):
        isoform_names = isoform_list[i]

        collapsed_expression_profile = np.zeros(expression_zeros.shape[0])
        for isoform_name in isoform_names.split(','):
            if isoform_name not in isoform_name_id_map.keys():
                isoform_id_key = -1
                print isoform_name
            else:
                isoform_id_key = isoform_name_id_map[isoform_name]
            collapsed_expression_profile += isoform_id_exp_map[isoform_id_key]

        expression_matrix_all[i, :] = collapsed_expression_profile

    np.save(expression_matrix_output_file, expression_matrix_all)
    return expression_matrix_all, isoform_list

def id_column_mapping(expression_file):
    sra_run_column_map = {}

    fr = open(expression_file, 'r')
    line = fr.readline()
    run_id = line.split('\t')[2: ]

    column = 0
    for run in run_id:
        run = run.split('_')[0]
        sra_run_column_map[run] = column
        column += 1

    fr.close()
    return sra_run_column_map

def study_run_id_mapping(sra_id_mapping_file):
    fr = open(sra_id_mapping_file, 'r')
    study_run_map = {}
    header = fr.readline()

    while True:
        line = fr.readline()
        if not line:
            break
        run = line.split('\t')[0]
        study = line.split('\t')[20]
        #print run, study

        if study not in study_run_map.keys() and study != '':
            study_run_map[study] = [run]
        elif study != '':
            value_tmp = study_run_map[study]
            value_tmp.append(run)
            study_run_map[study] = value_tmp

    fr.close()
    return study_run_map

def study_expression_matrix(expression_matrix_all, study_run_map, sra_run_column_map, isoform_list):
    for study_id in study_run_map.keys():
        if len(study_run_map[study_id]) >= 10:
            print study_id, len(study_run_map[study_id])
            # print study_map[studyID]
            exp_column = []
            for run_id in study_run_map[study_id]:
                exp_column.append(sra_run_column_map[run_id])

            study_exp_mat = expression_matrix_all[:, exp_column]

            study_matrix_file = '../data/raw_data/expression_data/expression_mat_' + study_id + '.csv'
            write_exp_matrix_csv(study_exp_mat, study_run_map[study_id], isoform_list, study_matrix_file)

def write_exp_matrix_csv(study_exp_mat, run_ids, isoform_list, exp_mat_ouput_file):
    fw = open(exp_mat_ouput_file, 'w')
    fw.write('isoform')
    for run_id in run_ids:
        fw.write(',' + run_id)
    fw.write('\n')
    for i in range(study_exp_mat.shape[0]):
        isoform_name = isoform_list[i].replace(',', '|')
        fw.write(isoform_name)
        for j in range(study_exp_mat[i].shape[0]):
            fw.write(',' + str(study_exp_mat[i][j]))
        fw.write('\n')
    fw.close()

if __name__=='__main__':
    iso_id_mapping_file = '../data/raw_data/expression_data/iso_id_mapping.txt'
    expression_file = '../data/raw_data/expression_data/isoform_expression_matrix.txt'
    expression_matrix_output_file = '../data/raw_data/expression_data/expression_mat_all.npy'
    isoform_train_list = '../data/id_lists/train_isoform_list.npy'
    isoform_test_list = '../data/id_lists/test_isoform_list.npy'
    expression_matrix_all, isoform_list = generate_full_expression_matrix(iso_id_mapping_file, expression_file, expression_matrix_output_file, isoform_train_list, isoform_test_list)
    print expression_matrix_all.shape

    sra_run_column_map = id_column_mapping(expression_file)

    sra_id_mapping_file = '../data/raw_data/expression_data/sra_study_id_mapping.txt'
    study_run_map = study_run_id_mapping(sra_id_mapping_file)

    study_expression_matrix(expression_matrix_all, study_run_map, sra_run_column_map, isoform_list)
