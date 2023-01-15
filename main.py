from executer import Execute
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)
def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='data/')
    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage')
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia5',
                        help="Available datasets: FactBench, BPDP, Dbpedia34k, Dbpedia5, Yago3K")
    parser.add_argument("--sub_dataset_path", type=str, default=None,
                        help="Available subpaths: bpdp/, domain/, domainrange/, mix/, property/, random/, range/,")
    parser.add_argument("--prop", type=str, default="None",
                        help="Available properties: architect, artist, author, commander, director, musicComposer, producer, None")
    parser.add_argument("--negative_triple_generation", type=str, default="False",
                        help="Available approaches: corrupted-triple-based, corrupted-time-based, False")
    parser.add_argument("--cmp_dataset", type=bool, default=True)
    parser.add_argument("--include_veracity", type=bool, default=False)

    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')

    # Models.
    parser.add_argument("--model", type=str, default='temporal-model',
                        help="Available models:full-Hybrid, KGE-only,text-only, text-KGE-Hybrid, path-only, text-path-Hybrid, KGE-path-Hybrid")
                        # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    parser.add_argument("--emb_type", type=str, default='dihedron',
                        help="Available TKG embeddings: dihedron")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=20)
    parser.add_argument("--min_num_epochs", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--val_batch_size', type=int, default=100)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)

    parser.add_argument("--preprocess", type=str, default='False',
                        help="Available options: False, Concat, SentEmb, TrainTestTriplesCreate")
    parser.add_argument("--ids_only", type=str, default=False)

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    args = argparse_default()
    if args.eval_dataset == "Yago3K":
        args.ids_only = True

    if args.preprocess != 'False':
        if args.preprocess == 'Concat':
            if args.eval_dataset == "Dbpedia34k":
                DBpedia34kDataset = True
                # ConcatEmbeddings(args, path_dataset_folder=args.path_dataset_folder,DBpedia34k=DBpedia34kDataset)
            print("concat done")
        elif args.preprocess == 'SentEmb':
            print("sentence vectors creation is done")
        elif args.preprocess == 'TrainTestTriplesCreation':
            print("Train and Test triples creation is complete")

    elif args.eval_dataset == "FactBench":
        datasets_class = [ "property/","range/", "domain/", "domainrange/", "mix/", "random/"]
        for cls in datasets_class:
            args = argparse_default()
                # args.max_num_epochs = epoc
                # args.model = mdl
            args.subpath = cls
            exc = Execute(args)
            exc.start()
                    # exit(1)
    elif args.eval_dataset=="BPDP":
        args = argparse_default()
            # args.max_num_epochs = epoc
            # args.model = mdl
        exc = Execute(args)
        exc.start()
                # exit(1)
    elif args.eval_dataset == "Dbpedia5" or args.eval_dataset == "Yago3K":
        exc = Execute(args)
        exc.start()

    else:
        print("Please specify a valid dataset")
        exit(1)
