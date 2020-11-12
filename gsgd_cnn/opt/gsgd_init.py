import torch





def recover_s_layer(value, idx, shape):
    assert value.device == idx.device
    if value.device.type == 'cpu':
        return torch.sparse.FloatTensor(idx, value, shape).to_dense()
    else:
        return torch.cuda.sparse.FloatTensor(idx, value, shape).to_dense()





def generate_eye_cnn(out_shape, in_shape, shape_2, shape_3, loc ):

    if loc == 'f':

        shape_2_center = shape_2 //2
        shape_3_center = shape_3 // 2

        ratio = out_shape // in_shape + 1

        out_idx = list(range(out_shape))
        in_idx = list(range(in_shape)) * ratio
        idx_tmp = list(zip(out_idx, in_idx))
        idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
        return(
            recover_s_layer(
                idx=idx,
                value=torch.ones(out_shape),
                shape=[out_shape , in_shape , shape_2 , shape_3])
        )

    elif loc == 'l':

        shape_2_center = shape_2 //2
        shape_3_center = shape_3 // 2

        ratio = in_shape // out_shape + 1

        out_idx = list(range(out_shape)) * ratio
        in_idx = list(range(in_shape))
        idx_tmp = list(zip(out_idx, in_idx))
        idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
        return(
            recover_s_layer(
                idx=idx,
                value=torch.ones(in_shape),
                shape=[out_shape , in_shape , shape_2 , shape_3])
        )
    else:

        shape_2_center = shape_2 //2
        shape_3_center = shape_3 // 2

        if in_shape > out_shape:
            ratio = in_shape // out_shape + 1

            out_idx = list(range(out_shape)) * ratio
            in_idx = list(range(in_shape))
        else:
            shape_2_center = shape_2 // 2
            shape_3_center = shape_3 // 2

            ratio = out_shape // in_shape + 1

            out_idx = list(range(out_shape))
            in_idx = list(range(in_shape)) * ratio

        idx_tmp = list(zip(out_idx, in_idx))
        idx = torch.LongTensor([(*x , shape_2_center , shape_3_center)  for x in idx_tmp]).transpose(0,1)
        return(
            recover_s_layer(
                idx=idx,
                value=torch.ones(max(out_shape, in_shape)),
                shape=[out_shape , in_shape , shape_2 , shape_3])
        )




def gsgd_init(model, args):
    

    params = list(model.parameters())

    ec_layer = model.ec_layer
    num_layer = len(params)
    mask = {'cnn': [],
            'mlp': []}


    for blocks_type, blocks_idx in ec_layer.items():
        if blocks_type == 'cnn':

            if len(blocks_idx) == 0:
                continue
            mask_cnn = []


            for block in blocks_idx:
                layer_mask = []
                num_layer_in_block = len(block)



                for ec_layer_idx, layer_idx in enumerate(block):
                    layer = params[layer_idx]
                    outcome, income,shape_2, shape_3 = layer.shape
                    if ec_layer_idx == 0:
                        loc = 'f'
                    elif ec_layer_idx == num_layer_in_block - 1:
                        loc = 'l'
                    else:
                        loc = 'm'

                    if loc == 'f':
                        red_init = args.first_red_init
                    else:
                        red_init = args.red_init

                    mask_tmp = generate_eye_cnn(outcome, income,shape_2, shape_3, loc).to(args.device)
                    layer.data = layer.data * (1 - mask_tmp)*args.blue_init  + mask_tmp *  red_init

    print('init done')
 
