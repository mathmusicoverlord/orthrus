"""
Generic script for visualizing a data set with the number of features changing over time.
"""

if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os
    import cv2
    import matplotlib.pyplot as plt

    # command line arguments
    parser = argparse.ArgumentParser("generic-visualization")
    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
                                             'setosa_versicolor_classify_species_svm',
                                             'setosa_versicolor_classify_species_svm_params.py'),
                        help='File path of containing the experimental parameters. Default is the Iris experiment.')

    parser.add_argument('--embedding',
                        type=str,
                        default='mds',
                        choices=['pca', 'mds'],
                        help='Method used to embed the data.')

    parser.add_argument('--dim',
                        type=int,
                        default=2,
                        choices=[1, 2, 3],
                        help='Dimension of embedding.')

    parser.add_argument('--increment',
                        type=int,
                        default=10,
                        help='Increment to increase the feature count by.')

    parser.add_argument('--fps',
                        type=int,
                        default=10,
                        help='Frames per second in the output movie.')


    args = parser.parse_args()

    # imports for script
    from sklearn.decomposition import PCA
    from umap import UMAP
    from datasci.manifold.mds import MDS
    from datasci.core.helper import module_from_path
    from datasci.decomposition.general import align_embedding
    from copy import deepcopy
    import pandas as pd

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    fig_dir = exp_params.FIG_DIR
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS
    feature_ids = exp_params.FEATURE_IDS # should be a list of ranked features (most important -> least important)

    # grab dimension
    if args.dim == 1:
        dim = 1
    if args.dim == 2:
        dim = 2
    elif args.dim == 3:
        dim = 3

    # grab embedding
    if args.embedding == 'umap':
        embedding = UMAP(n_components=dim)
    elif args.embedding == 'pca':
        embedding = PCA(n_components=dim, whiten=True, random_state=0)
    elif args.embedding == 'mds':
        embedding = MDS(n_components=dim)

    # get backend info
    if pd.api.types.infer_dtype(ds.metadata[class_attr]) == 'floating':
        palette = 'viridis'
    else:
        palette = 'bright'
    backend_args = dict(palette=palette, alpha=.7, mrkr_list=['o'], s=200)

    # generate video directory
    video_name = '_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower()])
    video_dir = os.path.join(fig_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    ds.path = video_dir

    # generate frames
    frame_number = 0
    adjusted_embedding = deepcopy(embedding)
    for num_features in range(0, len(feature_ids), args.increment):

        # check the number of features is above the dimension
        if num_features > args.dim:
            frame_number = frame_number + 1
            print(r"Generating frame number %d..." % (frame_number,))

            # visualize data
            _, embedding_vals = ds.visualize(embedding=adjusted_embedding,
                                             sample_ids=sample_ids,
                                             feature_ids=feature_ids[:num_features],
                                             attr=class_attr,
                                             subtitle=r'# Features = %d' % (num_features,),
                                             save=True,
                                             save_name=str(frame_number),
                                             block=False,
                                             **backend_args)

            adjusted_embedding = (align_embedding(embedding_vals))(deepcopy(embedding))
            plt.close()

    # grab images and sort them
    print("Generating movie file...")
    images = pd.Series([img for img in os.listdir(video_dir) if img.endswith(".png")])
    ordering = images.str.split('_').apply(lambda x: x[-1]).str.strip('.png').astype(int).argsort().values
    images = images.iloc[ordering].to_list()
    frame = cv2.imread(os.path.join(video_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(video_dir, 'movie.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(video_dir, image)))

    cv2.destroyAllWindows()
    video.release()
