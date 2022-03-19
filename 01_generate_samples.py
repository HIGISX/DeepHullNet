import argparse
from solver import  generate_concave_hull, generate_convex_hull, save_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generator')
    parser.add_argument(
        'problem',
        help='instance type to process.',
        choices=['convex', 'concave'],
    )

    args = parser.parse_args()
    gen_transfer = True
    gen_train = True

    if args.problem == 'convex':

        if gen_train == True:
            min_nodes = 50
            max_nodes = 50
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            # train instance
            n_train = 10000
            print('Starting generate convex hull samples...')
            train_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_train)
            train_file_name = save_data(train_datasets, cfg, name='train_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d train samples have been generated."%(n_train))
            # validation instance
            n_val = 2000
            vaild_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_val)
            val_file_name = save_data(vaild_datasets, cfg, name='val_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d val samples have been generated."%(n_val))
            # test instance
            n_test = 2000
            test_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_test)
            test_file_name = save_data(test_datasets, cfg, name='test_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d test samples have been generated."%(n_test))

        if gen_transfer == True:
            n_transfer = 100
            # small transfer instances
            min_nodes = 100
            max_nodes = 100
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_transfer)
            s_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_100_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d small transfer samples have been generated."%(n_transfer))

            # medium transfer instances
            min_nodes = 200
            max_nodes = 200
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_transfer)
            m_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_200_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d medium transfer samples have been generated."%(n_transfer))

            # large transfer instances
            min_nodes = 400
            max_nodes = 400
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_convex_hull(min_nodes, max_nodes, num_samples=n_transfer)
            l_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_400_dateset.json',
                                  data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d large transfer samples have been generated."%(n_transfer))

    elif args.problem == 'concave':
        if gen_train == True:
            min_nodes = 30
            max_nodes = 30
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            # train instance
            n_train = 10000
            print('Starting generate concave hull samples...')
            train_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_train)
            train_file_name = save_data(train_datasets, cfg, name='train_dateset.json',
                                        data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d train samples have been generated." % (n_train))
            # validation instance
            n_val = 2000
            vaild_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_val)
            val_file_name = save_data(vaild_datasets, cfg, name='val_dateset.json',
                                      data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d val samples have been generated." % (n_val))
            # test instance
            n_test = 2000
            test_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_test)
            test_file_name = save_data(test_datasets, cfg, name='test_dateset.json',
                                       data_dir=f'./dataset/{args.problem}/{min_nodes}-{max_nodes}/')
            print("%d test samples have been generated." % (n_test))

        if gen_transfer == True:
            n_transfer = 100
            # small transfer instances
            min_nodes = 100
            max_nodes = 100
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_transfer)
            s_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_100_dateset.json',
                                             data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d small transfer samples have been generated." % (n_transfer))

            # medium transfer instances
            min_nodes = 200
            max_nodes = 200
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_transfer)
            m_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_200_dateset.json',
                                             data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d medium transfer samples have been generated." % (n_transfer))

            # large transfer instances
            min_nodes = 400
            max_nodes = 400
            cfg = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
            transfer_datasets = generate_concave_hull(min_nodes, max_nodes, num_samples=n_transfer)
            l_transfer_file_name = save_data(transfer_datasets, cfg, name='transfer_400_dateset.json',
                                             data_dir=f'./dataset/{args.problem}/{max_nodes}/')
            print("%d large transfer samples have been generated." % (n_transfer))
    else:
        raise NotImplementedError
