
from keras_poison import find_stable_idx, find_vnb_idx_new, my_load_dataset, select_model, find_x_i


x_train, y_train, x_test, y_test = my_load_dataset(args.dataset)
model = select_model('vgg')


if args.slt_stb_ts_x:
    print('Selecting stable x by retraininretraingg 2 times using the same training data.')
    index = find_stable_idx(x_test, y_test, model)
    print('First 20 / {} index of stable x: \n{}'.format(len(index), index[:20]))
else:
    index = range(len(x_test))
    print('Selecting x in all testing data, First 20 index: \n{}'.format(index[:20]))

# decide which index
if args.slt_vnb_tr_x:
    index = find_vnb_idx_new(index, x_test, y_test, model)
    print('Successfully selected vulnerable data, First 20 index: \n{}'.format((index[:20])))

for idx in index:
    print('x idx:',idx)
    x = x_test[idx]

    if args.slt_x_i:
        x_i_idx = find_x_i(model, x, y_test[idx], x_train, y_train)
