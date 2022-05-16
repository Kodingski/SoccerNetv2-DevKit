import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import json
import torch
import random
import math

from dataset import SoccerNetClipsSportec #,SoccerNetClipsOld, SoccerNetClips, SoccerNetClipsTesting
from model import Model
from train import trainer, test, testSpotting
from loss import NLLLoss

#games currently not used
#game_list = [           
#            "SF_02ST_TSG_FCU",
#            "SF_02ST_BSC_WOB",
#            "SF_02ST_BOC_M05",
#            "SF_02ST_SGF_DSC",
#            ]


def main(args):
    
    print(args.pool)
    model_name = args.model_name
    num_classes = 3
    matches = [
                  "SF_02ST_RBL_VFB",
                  "SF_02ST_SGE_FCA",
                  'SF_02ST_SCF_BVB',  
                  "SF_02ST_TSG_FCU",
                  "SF_02ST_BSC_WOB",
                  "SF_02ST_BOC_M05",
                  "SF_02ST_SGF_DSC",
                  "SF_02ST_B04_BMG",
                  "SF_02ST_FCB_KOE"
                   ]
     
    labels = dict.fromkeys(matches)
    dict_event_sts = {'None':0, 'Play':1, 'TacklingGame':2, 'Throw-in':3}

    #num_classes = len(dict_event_sts)
    for match in matches:
        labels_match = json.load(open(os.path.join(args.path, 'labels', f'{match}_labels.json')))
        labels_encoded = np.zeros((len(labels_match['annotations']), num_classes+1))
        for annotation in labels_match["annotations"]:

            snippet_id = annotation['id']
            end_time = annotation["end"]
            event_timestamp = annotation["timestep"]
            #frame = framerate * ( seconds + 60 * minutes ) 

            event_type = annotation['type']
            label = dict_event_sts[event_type]

            labels_encoded[snippet_id,label] = 1 # that's my class
          
        labels[match] = labels_encoded
       
        

    clips_augmented = sorted(os.listdir(os.path.join(args.path, 'augmented_features')))
    clips_names = [clip.split('.')[0] for clip in clips_augmented]
    clips_unique = set(clips_names)
    #print(clips_unique)
    clips_unique = list(clips_unique)
    random.Random(42).shuffle(clips_unique)
    #print(clips_unique)

    
    n_clips = len(clips_unique)
    train, val, test_clips = clips_unique[:int(n_clips*0.7)], clips_unique[int(n_clips*0.7):int(n_clips*0.85)], clips_unique[int(n_clips*0.85):]
    
    

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClipsSportec(path=args.path, features=args.features, split=train, version=args.version, framerate=args.framerate, window_size=args.window_size, labels = labels)
        print(dataset_Train.game_feats.shape)
        print(dataset_Train.game_labels.shape)
        
    
        dataset_Valid = SoccerNetClipsSportec(path=args.path, features=args.features, split=val, version=args.version, framerate=args.framerate, window_size=args.window_size, labels = labels)
        dataset_Valid_metric  = SoccerNetClipsSportec(path=args.path, features=args.features, split=val, version=args.version, framerate=args.framerate, window_size=args.window_size, labels = labels)
    dataset_Test  = SoccerNetClipsSportec(path=args.path, features=args.features, split=test_clips, version=args.version, framerate=args.framerate, window_size=args.window_size, labels = labels)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][1].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    model = Model(weights=args.load_weights, input_size=args.feature_dim,
                  num_classes=dataset_Test.num_classes, window_size=args.window_size, 
                  vocab_size = args.vocab_size,
                  framerate=args.framerate, pool=args.pool, dropout = args.dropout).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))
    logging.info(f"Learning Rate: {args.LR}, Batch Size: {args.batch_size}")

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        
        #define class weights for NLLLoss:
        
        
        
        
        criterion = torch.nn.NLLLoss(weight= torch.tensor([(1/1910),(1/4177),(1/749),(1/220)], device = torch.device('cuda:0')))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=1e-4, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    #for split in args.split_test:
        #dataset_Test  = SoccerNetClipsSportec(path=args.path, features=args.features, split=test, version=args.version, framerate=args.framerate, window_size=args.window_size, labels = labels)

    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)

    
    test(test_loader, model, model_name,  'test')

    #a_mAP = results["a_mAP"]
    #a_mAP_per_class = results["a_mAP_per_class"]
    #a_mAP_visible = results["a_mAP_visible"]
    #a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
    #a_mAP_unshown = results["a_mAP_unshown"]
    #a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

    #logging.info("Best Performance at end of training ")
    #logging.info("a_mAP visibility all: " +  str(a_mAP))
    #logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
    #logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
    #logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
    #logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
    #logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

    return 

if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="snippets_features.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD++",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=2048,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=12.4,     help='Framerate of the input features' )
    parser.add_argument('--window_size', required=False, type=int,   default=2.5,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vocab_size',       required=False, type=int,   default=32, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=32,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=100, help='seed for reproducibility')
    parser.add_argument('--dropout', required=False, type=float, default=0.8, help='percentage to drop in dropout layer')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
