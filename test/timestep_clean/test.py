import numpy as np
import sys
sys.path.append('../../../torch1d')
from torch1d import inputs
from torch1d import *

if __name__ == "__main__":
    print("Checking if an error is accumulating over time integration..")
    args = inputs.parser.parse_args()
    config = inputs.InputParser(args.input_file)

    solverType = config.getInput(['system','type'], fallback='axial-torch')
    from axial_torch import AxialTorch
    solverDict = {'axial-torch': AxialTorch}

    solver1 = solverDict[solverType](config)
    solver2 = solverDict[solverType](config)

    icFilename = 'test.ic.h5'
    solver1.state.loadState(icFilename)
    solver2.state.loadState(icFilename)

    print("Checking if two solvers read the same ic..")
    if (np.any(solver1.state.conserved != solver2.state.conserved)):
        raise RuntimeError("Initial condition is not read properly!")

    solver1.timeIntegrator.step()
    step1 = np.copy(solver1.state.conserved)

    print("time after one timestep: ", solver1.timeIntegrator.time, solver1.state.time)
    solver2.state.conserved = np.copy(solver1.state.conserved)
    solver2.timeIntegrator.time = solver1.timeIntegrator.time
    solver2.state.time = solver1.timeIntegrator.time

    print("Copying the state after one timestep")
    if (np.any(solver1.state.conserved != solver2.state.conserved)):
        raise RuntimeError("Copy is not executed properly!")

    print("Two solvers must have the same solution after a timestep.")
    solver1.timeIntegrator.step()
    solver2.timeIntegrator.step()
    if (np.any(solver1.state.conserved != solver2.state.conserved)):
        raise RuntimeError("An error is accumulating over time integration!")
    else:
        print("Passed the test.")

    # solver2.timeIntegrator.trialState.update()
    #
    # solver1.timeIntegrator.step()
    # solver2.timeIntegrator.step()
    # print(np.any(solver1.state.conserved != solver2.state.conserved))
    #
    # solver2.timeIntegrator.trialState.update()
    # solver2.timeIntegrator.trialState.updateGradient()
    #
    # solver1.timeIntegrator.step()
    # solver2.timeIntegrator.step()
    # print(np.any(solver1.state.conserved != solver2.state.conserved))
