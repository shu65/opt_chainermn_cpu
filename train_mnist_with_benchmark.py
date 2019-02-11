#!/usr/bin/env python
from __future__ import print_function

import argparse
import cProfile
import io
import pstats
import multiprocessing

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import chainermn


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(784, n_units),  # n_in -> n_units
            l2=L.Linear(n_units, n_units),  # n_units -> n_units
            l3=L.Linear(n_units, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def dummy_func():
    pass


def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--benchmark', action='store_true',
                        help='benchmark mode')
    parser.add_argument('--benchmark-iteration', type=int, default=500,
                        help='the number of iterations when using benchmark mode')
    parser.add_argument('--cprofile', action='store_true', help='cprofile')
    args = parser.parse_args()

    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process(target=dummy_func, args=())
    p.start()
    p.join()

    # Prepare ChainerMN communicator.
    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    model = L.Classifier(MLP(args.unit, 10))
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob,
        n_prefetch=args.loaderjob)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, repeat=False, n_processes=args.loaderjob,
        n_prefetch=args.loaderjob)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    if args.benchmark:
        stop_trigger = (args.benchmark_iteration, 'iteration')
    else:
        stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        if args.benchmark:
            trainer.extend(extensions.LogReport(), trigger=stop_trigger)
        else:
            trainer.extend(extensions.dump_graph('main/loss'))
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
            trainer.extend(extensions.ProgressBar())

            if args.resume:
                chainer.serializers.load_npz(args.resume, trainer)

    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()

    trainer.run()

    if args.cprofile:
        pr.disable()
        s = io.StringIO()
        sort_by = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats()
        if comm.rank == 0:
            print(s.getvalue())
        pr.dump_stats('{0}/rank_{1}.cprofile'.format(args.out, comm.rank))


if __name__ == '__main__':
    main()
