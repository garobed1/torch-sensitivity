import numpy as np
import h5py
import argparse

testParser = argparse.ArgumentParser(description = "",
                                     formatter_class=argparse.RawTextHelpFormatter)
testParser.add_argument('solution_a', metavar='string', type=str,
                        help='filename for the first solution to compare.\n')
testParser.add_argument('solution_b', metavar='string', type=str,
                        help='filename for the second solution to compare.\n')

if __name__ == "__main__":
    args = testParser.parse_args()
    print("Solution A: %s" % args.solution_a)
    print("Solution B: %s" % args.solution_b)

    fA = h5py.File(args.solution_a, 'r')
    assert('conserved' in fA)
    assert('time' in fA['conserved'].attrs)
    assert('timestep' in fA['conserved'].attrs)
    dataA = fA['conserved'][...]

    fB = h5py.File(args.solution_b, 'r')
    assert('conserved' in fB)
    assert('time' in fB['conserved'].attrs)
    assert('timestep' in fB['conserved'].attrs)
    dataB = fB['conserved'][...]

    assert fA['conserved'].attrs['time'] == fB['conserved'].attrs['time'],                              \
    '\ntime A: %.5E\ntime B: %.5E' % (fA['conserved'].attrs['time'], fB['conserved'].attrs['time'])
    assert fA['conserved'].attrs['timestep'] == fB['conserved'].attrs['timestep'],                                      \
    '\ntimestep A: %08d\ntimestep B: %08d' % (fA['conserved'].attrs['timestep'], fB['conserved'].attrs['timestep'])

    assert(dataA.shape == dataB.shape)
    diff = np.abs((dataA - dataB) / np.maximum(1.0e-15 * np.ones(dataA.shape), np.abs(dataA)))
    assert np.amax(diff) < 1e-8, '\nMaximum difference: %.15E' % np.amax(diff)
