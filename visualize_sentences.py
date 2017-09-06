from argparse import ArgumentParser
from sklearn.manifold import TSNE

import plotly
import plotly.graph_objs as gobj
import numpy as np
import os

from siamese.model import LSTMSiameseNet, LSTMSiameseWord


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(dest='MODEL',
                        help='The path to a model\'s directory')
    parser.add_argument('--set', dest='SET', default='ignore',
                        help='The selected dataset key in the loader used for plotting.')
    parser.add_argument('-m', '--mode', dest='MODE', type=int, default=2,
                        help='1 for character-level model. 2 for word-level model. This needs to correspond with '
                             'the loaded model\'s type for the script to run correctly.')
    parser.add_argument('-f', '--file', dest='FILE', default='lstm.html',
                        help='The name HTML file to which the plot is saved. '
                             'The file will always be put in the plots/ directory.')
    return parser.parse_args()


def main(args):
    os.makedirs('plots', exist_ok=True)

    if args.MODE == 1:
        MClass = LSTMSiameseNet
    elif args.MODE == 2:
        MClass = LSTMSiameseWord
    else:
        raise ValueError('Invalid mode %d' % args.MODE)

    print('Loading model from %s ...' % args.MODEL, end=' ')
    model = MClass.load(args.MODEL, skip_gen=True)
    print('Done.')
    k = args.SET

    raw = model.loader.raw[k]
    raw_labels = model.loader.raw_label[k]
    color_pool = ['rgb(51, 80, 196)', 'rgb(232, 23, 23)', 'rgb(23, 229, 232)',
                  'rgb(23, 232, 65)', 'rgb(204, 232, 23)', 'rgb(51, 80, 196)',
                  'rgb(232, 128, 23)', 'rgb(173, 23, 232)', 'rgb(232, 86, 220)',
                  'rgb(22, 9, 97)', 'rgb(6, 71, 19)', 'rgb(143, 196, 57)',
                  'rgb(18, 196, 113)', 'rgb(114, 128, 135)', 'rgb(224, 177, 177)']
    label_color = {raw_labels[0]: color_pool.pop(0)}
    colors = [label_color[raw_labels[0]]]

    print('Predicting vectors...')
    mat = model.predict_sent_vector(raw[0])
    for i, s in enumerate(raw[1:]):
        print('\t%d/%d' % (i+2, len(raw)), end='\r')
        mat = np.vstack((mat, model.predict_sent_vector(s)))
        if raw_labels[i] not in label_color.keys():
            label_color[raw_labels[i]] = color_pool.pop(0)
        colors.append(label_color[raw_labels[i]])
    print('')

    print('Running PCA...', end=' ')
    tsne = TSNE(perplexity=30, n_components=2,
                init='pca', n_iter=5000, metric='manhattan')
    plot_matrix = tsne.fit_transform(mat)
    print('Done.')

    hovers = [' | '.join([sent, label]) for sent, label in zip(raw, raw_labels)]

    points = [gobj.Scatter(
        x=plot_matrix[:, 0],
        y=plot_matrix[:, 1],
        mode='markers',
        hovertext=hovers,
        marker=dict(
            size=10,
            color=colors,
        )
    )]

    layout = gobj.Layout(
        title='Siamese LSTM sentence embeddings',
        hovermode='closest', showlegend=False,
    )

    fig = gobj.Figure(data=points, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join('plots', args.FILE), auto_open=False)


if __name__ == '__main__':
    main(parse_arguments())
