import os


def resnet_ec_layer(num_blocks, block_type = 'BasicBlock'):
    if block_type == 'BasicBlock':
        return resnet_ec_layer_basicblock(num_blocks)
    elif block_type == 'Bottleneck':
        return resnet_ec_layer_bottleneck(num_blocks)
    else:
        raise NotImplementedError

def resnet_ec_layer_basicblock(num_blocks):

        t = 1
        conv_idx = []
        for i, n in enumerate(num_blocks):
            for j in range(n):
                conv_idx.append([t * 3, (t + 1) * 3])
                t += 2
                if j == 0 and i > 0:
                    t += 1
        return conv_idx

def resnet_ec_layer_bottleneck(num_blocks):
    t=1
    conv_idx=[ ]
    for i,n in enumerate(num_blocks):
        for j in range(n):
            conv_idx.append([t*3, (t+1)*3, (t+2)*3])
            t+=3
            if j == 0:
                t+=1
    return conv_idx




def plainnet_ec_layer(num_blocks):
    t = 1

    conv_idx = [ ]

    for i,n in enumerate(num_blocks):
        k=0
        tmp = []
        while k<n:
            tmp.append(t*3)
            t+=1
            k+=1
        conv_idx.append(tmp)

    return conv_idx



def plainnetnobn_ec_layer(num_blocks):
    t = 3

    conv_idx = [ ]

    for i,n in enumerate(num_blocks):
        k=0
        tmp = []
        while k<n:
            tmp.append(t )
            t+=1
            k+=1
        conv_idx.append(tmp)

    return conv_idx

